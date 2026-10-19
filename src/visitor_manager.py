"""Visitor management and auto-registration system for Katomaran Intelligent Face Tracker.

Maintains in-memory embedding gallery synchronized with SQLite database,
matches incoming embeddings against registered visitors, assigns new Face IDs,
and ensures unique visitor count integrity.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np

from src.database import DatabaseManager, get_utc_iso_timestamp

logger = logging.getLogger("katomaran.visitor_manager")


def compute_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two 1D feature vectors.

    Returns:
        Float similarity score between -1.0 and 1.0.
    """
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


class VisitorManager:
    """Orchestrates identity matching, auto-registration, and unique visitor metrics."""

    def __init__(
        self,
        db_manager: DatabaseManager,
        similarity_threshold: float = 0.45,
        id_prefix: str = "FACE_",
    ):
        self.db = db_manager
        self.similarity_threshold = float(similarity_threshold)
        self.id_prefix = id_prefix

        # In-memory gallery cache: face_id -> normalized embedding
        self._gallery: Dict[str, np.ndarray] = {}
        self.load_gallery_from_db()

    def load_gallery_from_db(self) -> None:
        """Load all registered embeddings from database into fast in-memory lookup cache."""
        embeddings = self.db.get_all_embeddings()
        self._gallery.clear()
        for fid, emb in embeddings.items():
            norm = np.linalg.norm(emb)
            if norm > 0:
                self._gallery[fid] = (emb / norm).astype(np.float32)
            else:
                self._gallery[fid] = emb.astype(np.float32)
        logger.info(
            f"Loaded {len(self._gallery)} registered visitor identities into matching gallery."
        )

    def find_best_match(
        self, embedding: np.ndarray
    ) -> Tuple[Optional[str], float]:
        """Compare an input embedding against all registered identities in the gallery.

        Args:
            embedding: 1-D numpy array of facial embedding.

        Returns:
            Tuple of (best_matching_face_id or None, max_similarity_score).
        """
        if embedding is None or len(self._gallery) == 0:
            return None, 0.0

        norm = np.linalg.norm(embedding)
        if norm == 0:
            return None, 0.0
        query_vec = (embedding / norm).astype(np.float32)

        best_id: Optional[str] = None
        best_sim: float = -1.0

        for face_id, gallery_vec in self._gallery.items():
            sim = float(np.dot(query_vec, gallery_vec))
            if sim > best_sim:
                best_sim = sim
                best_id = face_id

        if best_sim >= self.similarity_threshold and best_id is not None:
            return best_id, best_sim
        return None, max(0.0, best_sim)

    def register_or_recognize(
        self,
        embedding: np.ndarray,
        timestamp: Optional[str] = None,
    ) -> Tuple[str, bool, float]:
        """Match embedding against gallery; if recognized, return existing Face ID;
        if new identity, register in database and gallery, then return new Face ID.

        Args:
            embedding: 1-D facial feature vector.
            timestamp: Event timestamp ISO string.

        Returns:
            Tuple of:
                - face_id: Assigned or matched Face ID
                - is_new_registration: True if newly registered, False if recognized existing visitor
                - similarity_score: Similarity score with matched identity (1.0 if new registration)
        """
        ts = timestamp or get_utc_iso_timestamp()

        # Step 1: Check existing registered gallery
        matched_id, score = self.find_best_match(embedding)
        if matched_id is not None:
            logger.info(
                f"Recognized returning visitor {matched_id} with similarity score {score:.4f} "
                f"(threshold={self.similarity_threshold})."
            )
            # Update last_seen in database (without incrementing unique visitor count)
            self.db.update_visitor_activity(matched_id, timestamp=ts, increment_visit=False)
            return matched_id, False, score

        # Step 2: Confirmed new visitor - safely generate next sequential Face ID
        new_face_id = self.db.get_next_face_id(self.id_prefix)
        # Check against local gallery in case of uncommitted IDs
        while new_face_id in self._gallery:
            # Handle collision
            idx = int(new_face_id[len(self.id_prefix):]) + 1
            new_face_id = f"{self.id_prefix}{idx:03d}"

        # Normalize and store in cache
        norm = np.linalg.norm(embedding)
        normalized_vec = (embedding / norm).astype(np.float32) if norm > 0 else embedding
        self._gallery[new_face_id] = normalized_vec

        # Persist to database
        self.db.register_visitor(
            face_id=new_face_id,
            embedding=embedding,
            timestamp=ts,
            status="active",
        )
        logger.info(
            f"Auto-registered NEW visitor {new_face_id}. "
            f"Total unique visitors now: {len(self._gallery)}."
        )
        return new_face_id, True, 1.0

    @property
    def unique_visitor_count(self) -> int:
        """Return exact unique visitor count based on persistent registrations."""
        return self.db.get_unique_visitor_count()
