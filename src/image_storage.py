"""Cropped face image storage manager for Katomaran Intelligent Face Tracker.

Saves cropped face bounding boxes into organized date-based folders:
    logs/entries/YYYY-MM-DD/FACE_ID_TIMESTAMP.jpg
    logs/exits/YYYY-MM-DD/FACE_ID_TIMESTAMP.jpg
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger("katomaran.image_storage")


def sanitize_filename_component(text: str) -> str:
    """Sanitize string for safe usage in file paths."""
    return re.sub(r"[^\w\-.]", "_", text)


def crop_face(
    frame: np.ndarray,
    bbox: Sequence[int | float],
    margin_ratio: float = 0.15,
) -> Optional[np.ndarray]:
    """Safely crop face region from video frame with boundary clipping and optional margin.

    Args:
        frame: Full source image frame (H x W x C).
        bbox: [x1, y1, x2, y2] bounding box coordinates.
        margin_ratio: Margin to expand bounding box for context (e.g., 0.15 = 15%).

    Returns:
        Cropped BGR numpy array, or None if crop area is invalid.
    """
    if frame is None or frame.size == 0:
        return None

    h, w = frame.shape[:2]
    if len(bbox) < 4:
        return None

    x1, y1, x2, y2 = [int(round(c)) for c in bbox[:4]]

    # Compute margin
    bw = x2 - x1
    bh = y2 - y1
    if bw <= 0 or bh <= 0:
        return None

    mx = int(bw * margin_ratio)
    my = int(bh * margin_ratio)

    cx1 = max(0, x1 - mx)
    cy1 = max(0, y1 - my)
    cx2 = min(w, x2 + mx)
    cy2 = min(h, y2 + my)

    if cx2 <= cx1 or cy2 <= cy1:
        return None

    crop = frame[cy1:cy2, cx1:cx2]
    if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
        return None

    return crop.copy()


class ImageStorageManager:
    """Handles saving cropped face images to disk and generating portable relative paths."""

    def __init__(
        self,
        project_root: Path,
        entry_dir: str | Path = "logs/entries",
        exit_dir: str | Path = "logs/exits",
    ):
        self.project_root = Path(project_root).resolve()
        self.entry_dir = (
            Path(entry_dir)
            if Path(entry_dir).is_absolute()
            else (self.project_root / entry_dir)
        ).resolve()
        self.exit_dir = (
            Path(exit_dir)
            if Path(exit_dir).is_absolute()
            else (self.project_root / exit_dir)
        ).resolve()

        # Ensure base directories exist
        self.entry_dir.mkdir(parents=True, exist_ok=True)
        self.exit_dir.mkdir(parents=True, exist_ok=True)

    def save_face_image(
        self,
        frame: np.ndarray,
        bbox: Sequence[int | float],
        face_id: str,
        event_type: str,
        timestamp: Optional[datetime] = None,
    ) -> Optional[str]:
        """Crop and save face image to date-partitioned directory.

        Args:
            frame: Source video frame.
            bbox: [x1, y1, x2, y2] bounding box.
            face_id: Face identifier (e.g. FACE_001).
            event_type: 'ENTRY' or 'EXIT'.
            timestamp: Event timestamp (defaults to current UTC datetime).

        Returns:
            Relative path string (forward slashes) from project_root to saved image,
            or None if saving failed.
        """
        crop = crop_face(frame, bbox)
        if crop is None:
            logger.error(f"Cannot save face image: invalid crop for {face_id} with bbox {bbox}.")
            return None

        dt = timestamp or datetime.now(timezone.utc)
        date_str = dt.strftime("%Y-%m-%d")
        time_str = dt.strftime("%Y%m%d_%H%M%S_%f")

        base_dir = self.entry_dir if event_type.upper() == "ENTRY" else self.exit_dir
        dest_dir = base_dir / date_str
        dest_dir.mkdir(parents=True, exist_ok=True)

        safe_face_id = sanitize_filename_component(face_id)
        filename = f"{safe_face_id}_{time_str}.jpg"
        target_path = dest_dir / filename

        # Avoid collision if microsecond collision occurs
        counter = 1
        while target_path.exists():
            target_path = dest_dir / f"{safe_face_id}_{time_str}_{counter}.jpg"
            counter += 1

        try:
            success = cv2.imwrite(str(target_path), crop)
            if not success or not target_path.exists():
                logger.error(f"Failed to write image to disk: {target_path}")
                return None

            # Compute relative path using forward slashes for cross-platform portability
            rel_path = target_path.relative_to(self.project_root).as_posix()
            logger.debug(f"Saved {event_type} crop: {rel_path}")
            return rel_path
        except Exception as e:
            logger.error(f"Filesystem error saving face image for {face_id}: {e}", exc_info=True)
            return None
