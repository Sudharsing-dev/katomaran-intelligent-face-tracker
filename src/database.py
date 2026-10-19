"""SQLite persistence layer for Katomaran Intelligent Face Tracker.

Manages database schema initialization, visitor identity registration,
event records, and embedding vector serialization/validation.
"""

from __future__ import annotations

import contextlib
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("katomaran.database")


def serialize_embedding(embedding: np.ndarray) -> bytes:
    """Serialize a 1D numpy array embedding into a binary blob.

    Args:
        embedding: 1-D numpy array of float values (e.g. 512-dim ArcFace).

    Returns:
        Raw bytes suitable for SQLite BLOB storage.
    """
    if embedding is None:
        raise ValueError("Cannot serialize None embedding.")
    arr = np.asarray(embedding, dtype=np.float32)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"Embedding must be a non-empty 1D array. Got shape {arr.shape}.")
    if np.isnan(arr).any() or np.isinf(arr).any():
        raise ValueError("Embedding contains NaN or Infinite values.")
    return arr.tobytes()


def deserialize_embedding(blob: bytes) -> np.ndarray:
    """Deserialize a binary blob back into a 1D float32 numpy array.

    Args:
        blob: Raw bytes from SQLite BLOB column.

    Returns:
        1-D numpy array with float32 values.
    """
    if not blob or len(blob) == 0:
        raise ValueError("Cannot deserialize empty embedding blob.")
    arr = np.frombuffer(blob, dtype=np.float32)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"Corrupted embedding blob. Parsed shape {arr.shape}.")
    if np.isnan(arr).any() or np.isinf(arr).any():
        raise ValueError("Deserialized embedding contains NaN or Infinite values.")
    return arr


def get_utc_iso_timestamp() -> str:
    """Return current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


class DatabaseManager:
    """Manages SQLite connections, transactions, visitor records, and event storage."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize_schema()

    @contextlib.contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager providing an active SQLite connection with foreign keys enabled."""
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=30.0,
            check_same_thread=False,
            isolation_level=None,  # Autocommit mode, explicit transactions via BEGIN/COMMIT
        )
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.execute("PRAGMA journal_mode = WAL;")  # High concurrency & crash recovery
            yield conn
        finally:
            conn.close()

    def initialize_schema(self) -> None:
        """Create database tables and indexes if they do not already exist."""
        logger.info(f"Initializing database schema at '{self.db_path}'...")
        with self.get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            try:
                # Visitors table
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS visitors (
                        face_id TEXT PRIMARY KEY,
                        first_seen TEXT NOT NULL,
                        last_seen TEXT NOT NULL,
                        embedding BLOB NOT NULL,
                        total_visits INTEGER NOT NULL DEFAULT 1,
                        status TEXT DEFAULT 'active'
                    );
                    """
                )

                # Events table
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS events (
                        event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        face_id TEXT NOT NULL,
                        event_type TEXT NOT NULL CHECK(event_type IN ('ENTRY', 'EXIT')),
                        timestamp TEXT NOT NULL,
                        image_path TEXT NOT NULL,
                        track_id INTEGER,
                        FOREIGN KEY (face_id) REFERENCES visitors(face_id) ON DELETE CASCADE
                    );
                    """
                )

                # Performance indexes
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_events_face_id ON events(face_id);"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_visitors_last_seen ON visitors(last_seen);"
                )

                conn.execute("COMMIT;")
                logger.info("Database schema initialized successfully.")
            except Exception as e:
                conn.execute("ROLLBACK;")
                logger.error(f"Failed to initialize database schema: {e}", exc_info=True)
                raise

    def get_unique_visitor_count(self) -> int:
        """Return the count of unique registered visitors."""
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) AS count FROM visitors;")
            row = cursor.fetchone()
            return row["count"] if row else 0

    def register_visitor(
        self,
        face_id: str,
        embedding: np.ndarray,
        timestamp: Optional[str] = None,
        status: str = "active",
    ) -> bool:
        """Insert a newly registered visitor into the database.

        Args:
            face_id: Unique face identifier (e.g. FACE_001).
            embedding: 1-D numpy array.
            timestamp: ISO 8601 timestamp string (defaults to current UTC).
            status: Status metadata ('active', 'exited', etc.).

        Returns:
            True if inserted successfully, False if duplicate or failed.
        """
        ts = timestamp or get_utc_iso_timestamp()
        blob = serialize_embedding(embedding)

        with self.get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            try:
                conn.execute(
                    """
                    INSERT INTO visitors (face_id, first_seen, last_seen, embedding, total_visits, status)
                    VALUES (?, ?, ?, ?, 1, ?);
                    """,
                    (face_id, ts, ts, blob, status),
                )
                conn.execute("COMMIT;")
                logger.info(f"Registered new visitor {face_id} in database at {ts}.")
                return True
            except sqlite3.IntegrityError as e:
                conn.execute("ROLLBACK;")
                logger.warning(f"Visitor {face_id} already exists or integrity violated: {e}")
                return False
            except Exception as e:
                conn.execute("ROLLBACK;")
                logger.error(f"Failed to register visitor {face_id}: {e}", exc_info=True)
                raise

    def update_visitor_activity(
        self,
        face_id: str,
        timestamp: Optional[str] = None,
        increment_visit: bool = False,
        status: Optional[str] = None,
    ) -> bool:
        """Update last_seen timestamp and optionally increment total_visits count.

        Args:
            face_id: Unique face identifier.
            timestamp: ISO 8601 timestamp.
            increment_visit: If True, increments total_visits by 1 (e.g., returning visit).
            status: New status string (or None to leave unchanged).
        """
        ts = timestamp or get_utc_iso_timestamp()
        with self.get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            try:
                if increment_visit:
                    if status is not None:
                        conn.execute(
                            """
                            UPDATE visitors
                            SET last_seen = ?, total_visits = total_visits + 1, status = ?
                            WHERE face_id = ?;
                            """,
                            (ts, status, face_id),
                        )
                    else:
                        conn.execute(
                            """
                            UPDATE visitors
                            SET last_seen = ?, total_visits = total_visits + 1
                            WHERE face_id = ?;
                            """,
                            (ts, face_id),
                        )
                else:
                    if status is not None:
                        conn.execute(
                            """
                            UPDATE visitors
                            SET last_seen = ?, status = ?
                            WHERE face_id = ?;
                            """,
                            (ts, status, face_id),
                        )
                    else:
                        conn.execute(
                            """
                            UPDATE visitors
                            SET last_seen = ?
                            WHERE face_id = ?;
                            """,
                            (ts, face_id),
                        )
                conn.execute("COMMIT;")
                return True
            except Exception as e:
                conn.execute("ROLLBACK;")
                logger.error(f"Failed to update activity for visitor {face_id}: {e}")
                return False

    def record_event(
        self,
        face_id: str,
        event_type: str,
        image_path: str,
        timestamp: Optional[str] = None,
        track_id: Optional[int] = None,
    ) -> Optional[int]:
        """Record an ENTRY or EXIT event in the database.

        Args:
            face_id: The visitor's Face ID.
            event_type: 'ENTRY' or 'EXIT'.
            image_path: Relative path to saved cropped face image.
            timestamp: ISO 8601 timestamp.
            track_id: Temporary tracking identifier.

        Returns:
            The newly inserted event_id, or None if failed.
        """
        if event_type not in ("ENTRY", "EXIT"):
            raise ValueError(f"Invalid event_type: {event_type}. Must be 'ENTRY' or 'EXIT'.")

        ts = timestamp or get_utc_iso_timestamp()
        with self.get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            try:
                cursor = conn.execute(
                    """
                    INSERT INTO events (face_id, event_type, timestamp, image_path, track_id)
                    VALUES (?, ?, ?, ?, ?);
                    """,
                    (face_id, event_type, ts, image_path, track_id),
                )
                event_id = cursor.lastrowid

                # Update visitor status accordingly
                new_status = "active" if event_type == "ENTRY" else "exited"
                conn.execute(
                    "UPDATE visitors SET last_seen = ?, status = ? WHERE face_id = ?;",
                    (ts, new_status, face_id),
                )
                conn.execute("COMMIT;")
                logger.info(
                    f"Recorded event #{event_id}: {event_type} for {face_id} "
                    f"(track_id={track_id}) image='{image_path}'"
                )
                return event_id
            except Exception as e:
                conn.execute("ROLLBACK;")
                logger.error(f"Failed to record {event_type} event for {face_id}: {e}")
                return None

    def get_all_embeddings(self) -> Dict[str, np.ndarray]:
        """Retrieve all registered embeddings from the database for matching.

        Returns:
            Dictionary mapping face_id -> 1D numpy array.
        """
        gallery: Dict[str, np.ndarray] = {}
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT face_id, embedding FROM visitors;")
            for row in cursor.fetchall():
                try:
                    gallery[row["face_id"]] = deserialize_embedding(row["embedding"])
                except Exception as e:
                    logger.warning(f"Error deserializing embedding for {row['face_id']}: {e}")
        return gallery

    def get_visitor(self, face_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a single visitor record by face_id."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT face_id, first_seen, last_seen, total_visits, status FROM visitors WHERE face_id = ?;",
                (face_id,),
            )
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None

    def get_all_visitors(self) -> List[Dict[str, Any]]:
        """Fetch all visitor records ordered by first_seen."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT face_id, first_seen, last_seen, total_visits, status FROM visitors ORDER BY first_seen ASC;"
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_events(
        self,
        face_id: Optional[str] = None,
        event_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Retrieve events with optional filters."""
        query = "SELECT event_id, face_id, event_type, timestamp, image_path, track_id FROM events WHERE 1=1"
        params: List[Any] = []
        if face_id:
            query += " AND face_id = ?"
            params.append(face_id)
        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)
        query += " ORDER BY event_id DESC LIMIT ? OFFSET ?;"
        params.extend([limit, offset])

        with self.get_connection() as conn:
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def get_event_counts(self) -> Dict[str, int]:
        """Return counts of total events grouped by type."""
        counts = {"ENTRY": 0, "EXIT": 0}
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT event_type, COUNT(*) as cnt FROM events GROUP BY event_type;")
            for row in cursor.fetchall():
                counts[row["event_type"]] = row["cnt"]
        return counts

    def get_next_face_id(self, prefix: str = "FACE_") -> str:
        """Generate next sequential Face ID safely based on maximum existing index."""
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT face_id FROM visitors WHERE face_id LIKE ?;", (f"{prefix}%",))
            rows = cursor.fetchall()
            max_idx = 0
            for r in rows:
                val = r["face_id"]
                if val.startswith(prefix):
                    suffix = val[len(prefix):]
                    if suffix.isdigit():
                        max_idx = max(max_idx, int(suffix))
            return f"{prefix}{max_idx + 1:03d}"
