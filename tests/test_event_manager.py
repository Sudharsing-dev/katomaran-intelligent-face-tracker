"""Unit tests for track lifecycle state machine and event management."""

import numpy as np
import pytest
from pathlib import Path

from src.database import DatabaseManager
from src.event_manager import EventManager, LifecycleState
from src.image_storage import ImageStorageManager


@pytest.fixture
def event_env(tmp_path: Path):
    db = DatabaseManager(tmp_path / "events_test.db")
    storage = ImageStorageManager(tmp_path)
    # Register dummy visitor
    emb = np.random.randn(512).astype(np.float32)
    db.register_visitor("FACE_001", emb)
    db.register_visitor("FACE_002", emb)

    event_mgr = EventManager(
        db_manager=db,
        image_storage=storage,
        exit_grace_frames=5,  # Short grace period for fast testing
        end_of_stream_policy="mark_active",
    )
    return db, storage, event_mgr


def test_single_entry_event_per_track(event_env):
    db, storage, mgr = event_env
    dummy_frame = np.ones((100, 100, 3), dtype=np.uint8) * 150
    bbox = [10, 10, 60, 60]

    # Frame 1: Track 1 initialized with FACE_001
    mgr.update_track_detection(1, bbox, dummy_frame, face_id="FACE_001")
    assert db.get_event_counts()["ENTRY"] == 1
    assert mgr.tracks[1].state == LifecycleState.ACTIVE

    # Frame 2-10: Track 1 continues visible in same location
    for _ in range(9):
        mgr.update_track_detection(1, bbox, dummy_frame, face_id="FACE_001")
        # Should still be exactly 1 ENTRY event!
        assert db.get_event_counts()["ENTRY"] == 1


def test_grace_period_and_exit_confirmation(event_env):
    db, storage, mgr = event_env
    dummy_frame = np.ones((100, 100, 3), dtype=np.uint8) * 150
    bbox = [10, 10, 60, 60]

    # Frame 1: Person appears
    mgr.update_track_detection(1, bbox, dummy_frame, face_id="FACE_001")
    assert db.get_event_counts()["ENTRY"] == 1
    assert db.get_event_counts()["EXIT"] == 0

    # Frames 2-4: Person disappears (missing for 3 frames < grace of 5)
    for _ in range(3):
        exited = mgr.process_missing_tracks(active_track_ids_in_frame=[])
        assert len(exited) == 0
        assert mgr.tracks[1].state == LifecycleState.EXIT_PENDING
        assert db.get_event_counts()["EXIT"] == 0

    # Frame 5: Person reappears within grace period!
    mgr.update_track_detection(1, bbox, dummy_frame, face_id="FACE_001")
    assert mgr.tracks[1].state == LifecycleState.ACTIVE
    assert db.get_event_counts()["ENTRY"] == 1
    assert db.get_event_counts()["EXIT"] == 0

    # Now person leaves completely: missing for 5 frames
    for _ in range(5):
        mgr.process_missing_tracks(active_track_ids_in_frame=[])

    # Exit should now be confirmed!
    assert mgr.tracks[1].state == LifecycleState.EXITED
    assert db.get_event_counts()["EXIT"] == 1


def test_returning_visitor_new_lifecycle(event_env):
    db, storage, mgr = event_env
    frame = np.ones((100, 100, 3), dtype=np.uint8) * 100
    bbox = [20, 20, 70, 70]

    # Visit 1: track #1
    mgr.update_track_detection(1, bbox, frame, face_id="FACE_001")
    # Person exits
    for _ in range(5):
        mgr.process_missing_tracks([])
    assert db.get_event_counts()["ENTRY"] == 1
    assert db.get_event_counts()["EXIT"] == 1

    # Visit 2: Person returns later as track #2, but recognized as same FACE_001
    mgr.update_track_detection(2, bbox, frame, face_id="FACE_001")
    assert db.get_event_counts()["ENTRY"] == 2
    assert db.get_event_counts()["EXIT"] == 1

    # Person exits again
    for _ in range(5):
        mgr.process_missing_tracks([])
    assert db.get_event_counts()["ENTRY"] == 2
    assert db.get_event_counts()["EXIT"] == 2


def test_multiple_independent_tracks(event_env):
    db, storage, mgr = event_env
    frame = np.ones((100, 100, 3), dtype=np.uint8) * 100

    # Track 1: FACE_001
    mgr.update_track_detection(1, [10, 10, 40, 40], frame, face_id="FACE_001")
    # Track 2: FACE_002
    mgr.update_track_detection(2, [50, 50, 90, 90], frame, face_id="FACE_002")

    assert db.get_event_counts()["ENTRY"] == 2

    # Track 1 leaves, Track 2 remains visible
    for _ in range(5):
        mgr.update_track_detection(2, [50, 50, 90, 90], frame, face_id="FACE_002")
        mgr.process_missing_tracks(active_track_ids_in_frame=[2])

    assert mgr.tracks[1].state == LifecycleState.EXITED
    assert mgr.tracks[2].state == LifecycleState.ACTIVE
    assert db.get_event_counts()["EXIT"] == 1
