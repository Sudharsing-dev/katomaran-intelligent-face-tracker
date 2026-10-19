"""Unit tests verifying unique visitor counting logic."""

import numpy as np
import pytest
from pathlib import Path

from src.database import DatabaseManager
from src.event_manager import EventManager
from src.image_storage import ImageStorageManager
from src.visitor_manager import VisitorManager


def test_unique_visitor_counting_guarantee(tmp_path: Path):
    db = DatabaseManager(tmp_path / "count_test.db")
    storage = ImageStorageManager(tmp_path)
    visitor_mgr = VisitorManager(db, similarity_threshold=0.50)
    event_mgr = EventManager(db, storage, exit_grace_frames=3)

    # Initial state: 0 unique visitors
    assert visitor_mgr.unique_visitor_count == 0
    assert db.get_unique_visitor_count() == 0

    frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
    emb_person_a = np.random.randn(512).astype(np.float32)

    # Person A appears in frame 1 -> NEW registration -> count = 1
    fid_a, is_new_a, _ = visitor_mgr.register_or_recognize(emb_person_a)
    event_mgr.update_track_detection(1, [10, 10, 50, 50], frame, face_id=fid_a)
    assert is_new_a is True
    assert visitor_mgr.unique_visitor_count == 1
    assert db.get_unique_visitor_count() == 1

    # Person A is tracked for 100 continuous frames -> count MUST remain 1
    for f in range(2, 102):
        # Even if recognition is re-evaluated with minor noise:
        noise = np.random.randn(512).astype(np.float32) * 0.02
        fid_check, is_new_check, _ = visitor_mgr.register_or_recognize(emb_person_a + noise)
        assert fid_check == fid_a
        assert is_new_check is False
        event_mgr.update_track_detection(1, [10, 10, 50, 50], frame, face_id=fid_check)
        assert visitor_mgr.unique_visitor_count == 1

    # Person A leaves the scene
    for _ in range(4):
        event_mgr.process_missing_tracks([])
    assert visitor_mgr.unique_visitor_count == 1

    # Person B appears -> NEW registration -> count = 2
    emb_person_b = np.random.randn(512).astype(np.float32)
    # Ensure orthogonality to prevent accidental high similarity
    emb_person_b = emb_person_b - (np.dot(emb_person_b, emb_person_a) / np.dot(emb_person_a, emb_person_a)) * emb_person_a
    fid_b, is_new_b, _ = visitor_mgr.register_or_recognize(emb_person_b)
    event_mgr.update_track_detection(2, [20, 20, 60, 60], frame, face_id=fid_b)
    assert is_new_b is True
    assert fid_b != fid_a
    assert visitor_mgr.unique_visitor_count == 2
    assert db.get_unique_visitor_count() == 2

    # Person A returns later (as track 3) -> Recognized -> count MUST remain 2
    fid_a_return, is_new_a_return, score = visitor_mgr.register_or_recognize(emb_person_a)
    assert fid_a_return == fid_a
    assert is_new_a_return is False
    event_mgr.update_track_detection(3, [15, 15, 55, 55], frame, face_id=fid_a_return)
    assert visitor_mgr.unique_visitor_count == 2
    assert db.get_unique_visitor_count() == 2

    # Verify event counts: 3 entries total (A twice, B once) but strictly 2 unique visitors!
    event_counts = db.get_event_counts()
    assert event_counts["ENTRY"] == 3
    assert db.get_unique_visitor_count() == 2
