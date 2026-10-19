"""Unit tests for SQLite database schema, embedding persistence, and transactions."""

import sqlite3
import numpy as np
import pytest
from pathlib import Path

from src.database import (
    DatabaseManager,
    deserialize_embedding,
    serialize_embedding,
)


@pytest.fixture
def test_db(tmp_path: Path) -> DatabaseManager:
    db_file = tmp_path / "test_visitors.db"
    return DatabaseManager(db_file)


def test_embedding_serialization_roundtrip():
    original = np.random.randn(512).astype(np.float32)
    blob = serialize_embedding(original)
    assert isinstance(blob, bytes)
    assert len(blob) == 512 * 4  # 512 float32 = 2048 bytes

    recovered = deserialize_embedding(blob)
    np.testing.assert_allclose(original, recovered, rtol=1e-5)


def test_embedding_nan_inf_rejection():
    bad_emb = np.array([1.0, np.nan, 2.0], dtype=np.float32)
    with pytest.raises(ValueError, match="NaN"):
        serialize_embedding(bad_emb)

    inf_emb = np.array([1.0, np.inf, 2.0], dtype=np.float32)
    with pytest.raises(ValueError, match="Infinite"):
        serialize_embedding(inf_emb)


def test_visitor_registration_and_count(test_db: DatabaseManager):
    assert test_db.get_unique_visitor_count() == 0

    emb1 = np.random.randn(512).astype(np.float32)
    success = test_db.register_visitor("FACE_001", emb1)
    assert success is True
    assert test_db.get_unique_visitor_count() == 1

    # Duplicate registration of same face_id should fail
    duplicate_success = test_db.register_visitor("FACE_001", emb1)
    assert duplicate_success is False
    assert test_db.get_unique_visitor_count() == 1


def test_foreign_key_enforcement(test_db: DatabaseManager):
    # Attempting to record an event for non-existent FACE_999 must fail
    res = test_db.record_event(
        face_id="FACE_999",
        event_type="ENTRY",
        image_path="logs/entries/fake.jpg",
    )
    assert res is None  # Caught and returned None due to foreign key failure


def test_record_entry_and_exit_events(test_db: DatabaseManager):
    emb = np.random.randn(512).astype(np.float32)
    test_db.register_visitor("FACE_001", emb)

    # ENTRY event
    entry_id = test_db.record_event(
        face_id="FACE_001",
        event_type="ENTRY",
        image_path="logs/entries/2026-10-17/FACE_001_1.jpg",
        track_id=1,
    )
    assert entry_id is not None

    # EXIT event
    exit_id = test_db.record_event(
        face_id="FACE_001",
        event_type="EXIT",
        image_path="logs/exits/2026-10-17/FACE_001_exit.jpg",
        track_id=1,
    )
    assert exit_id is not None
    assert exit_id > entry_id

    # Verify counts
    counts = test_db.get_event_counts()
    assert counts["ENTRY"] == 1
    assert counts["EXIT"] == 1

    # Verify visitor status was updated
    vis = test_db.get_visitor("FACE_001")
    assert vis["status"] == "exited"


def test_get_next_face_id(test_db: DatabaseManager):
    assert test_db.get_next_face_id() == "FACE_001"
    emb = np.random.randn(512).astype(np.float32)
    test_db.register_visitor("FACE_001", emb)
    assert test_db.get_next_face_id() == "FACE_002"
