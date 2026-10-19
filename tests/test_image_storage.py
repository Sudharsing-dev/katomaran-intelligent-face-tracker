"""Unit tests for face crop extraction, boundary clipping, and date-based storage."""

from datetime import datetime, timezone
from pathlib import Path
import cv2
import numpy as np
import pytest

from src.image_storage import ImageStorageManager, crop_face


def test_crop_face_clipping():
    # 200x200 black image with a 50x50 white square in center
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    frame[50:100, 50:100] = 255

    # Test bounding box partially outside image boundaries
    crop = crop_face(frame, [-20, -10, 80, 70], margin_ratio=0.0)
    assert crop is not None
    # Coordinates should be clipped to [0, 0, 80, 70]
    assert crop.shape == (70, 80, 3)


def test_crop_face_invalid():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    # Inverted box x2 < x1
    crop = crop_face(frame, [80, 80, 50, 50])
    assert crop is None

    # Empty frame
    assert crop_face(None, [10, 10, 50, 50]) is None


def test_save_entry_and_exit_image(tmp_path: Path):
    storage = ImageStorageManager(
        project_root=tmp_path,
        entry_dir="logs/entries",
        exit_dir="logs/exits",
    )

    frame = np.ones((200, 200, 3), dtype=np.uint8) * 120
    bbox = [40, 40, 120, 120]
    now = datetime(2026, 10, 17, 12, 0, 0, tzinfo=timezone.utc)

    # Save ENTRY image
    entry_path = storage.save_face_image(frame, bbox, "FACE_001", "ENTRY", timestamp=now)
    assert entry_path is not None
    assert entry_path.startswith("logs/entries/2026-10-17/FACE_001_")
    assert (tmp_path / entry_path).exists()

    # Save EXIT image
    exit_path = storage.save_face_image(frame, bbox, "FACE_001", "EXIT", timestamp=now)
    assert exit_path is not None
    assert exit_path.startswith("logs/exits/2026-10-17/FACE_001_")
    assert (tmp_path / exit_path).exists()


def test_save_image_collision_prevention(tmp_path: Path):
    storage = ImageStorageManager(project_root=tmp_path)
    frame = np.ones((100, 100, 3), dtype=np.uint8) * 200
    bbox = [10, 10, 50, 50]
    fixed_time = datetime(2026, 10, 17, 14, 30, 0, 123456, tzinfo=timezone.utc)

    path1 = storage.save_face_image(frame, bbox, "FACE_002", "ENTRY", timestamp=fixed_time)
    path2 = storage.save_face_image(frame, bbox, "FACE_002", "ENTRY", timestamp=fixed_time)

    assert path1 != path2
    assert (tmp_path / path1).exists()
    assert (tmp_path / path2).exists()
