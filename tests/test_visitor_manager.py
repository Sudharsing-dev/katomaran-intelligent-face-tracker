"""Unit tests for visitor management, cosine matching, and auto-registration."""

import numpy as np
import pytest
from pathlib import Path

from src.database import DatabaseManager
from src.visitor_manager import VisitorManager, compute_cosine_similarity


@pytest.fixture
def visitor_setup(tmp_path: Path):
    db_file = tmp_path / "visitors_test.db"
    db = DatabaseManager(db_file)
    manager = VisitorManager(db_manager=db, similarity_threshold=0.50)
    return db, manager


def test_cosine_similarity_math():
    v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v3 = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    assert pytest.approx(compute_cosine_similarity(v1, v2)) == 1.0
    assert pytest.approx(compute_cosine_similarity(v1, v3)) == 0.0


def test_first_visitor_registration(visitor_setup):
    db, manager = visitor_setup
    emb1 = np.random.randn(512).astype(np.float32)

    face_id, is_new, score = manager.register_or_recognize(emb1)
    assert face_id == "FACE_001"
    assert is_new is True
    assert score == 1.0
    assert manager.unique_visitor_count == 1


def test_returning_visitor_recognition(visitor_setup):
    db, manager = visitor_setup
    base_emb = np.random.randn(512).astype(np.float32)

    # First registration
    face_id1, is_new1, _ = manager.register_or_recognize(base_emb)
    assert face_id1 == "FACE_001"
    assert is_new1 is True
    assert manager.unique_visitor_count == 1

    # Simulate same person in subsequent frame with slight noise (cosine similarity ~ 0.95)
    noise = np.random.randn(512).astype(np.float32) * 0.05
    similar_emb = base_emb + noise

    face_id2, is_new2, score2 = manager.register_or_recognize(similar_emb)
    assert face_id2 == "FACE_001"  # Matched existing!
    assert is_new2 is False
    assert score2 > 0.80
    assert manager.unique_visitor_count == 1  # Did NOT increment!


def test_distinct_second_visitor_registration(visitor_setup):
    db, manager = visitor_setup
    # Orthogonal embeddings
    emb1 = np.zeros(512, dtype=np.float32)
    emb1[0] = 1.0

    emb2 = np.zeros(512, dtype=np.float32)
    emb2[1] = 1.0

    fid1, is_new1, _ = manager.register_or_recognize(emb1)
    assert fid1 == "FACE_001"
    assert is_new1 is True

    fid2, is_new2, _ = manager.register_or_recognize(emb2)
    assert fid2 == "FACE_002"
    assert is_new2 is True
    assert manager.unique_visitor_count == 2


def test_gallery_persistence_across_restarts(tmp_path: Path):
    db_file = tmp_path / "restart_test.db"
    db = DatabaseManager(db_file)
    mgr1 = VisitorManager(db, similarity_threshold=0.5)

    emb = np.random.randn(512).astype(np.float32)
    mgr1.register_or_recognize(emb)
    assert mgr1.unique_visitor_count == 1

    # New manager instance with same DB
    mgr2 = VisitorManager(db, similarity_threshold=0.5)
    assert mgr2.unique_visitor_count == 1

    # Recognize using mgr2
    fid, is_new, score = mgr2.register_or_recognize(emb)
    assert fid == "FACE_001"
    assert is_new is False
    assert score > 0.99
