"""Multi-face object tracking module for Katomaran Intelligent Face Tracker.

Implements a robust multi-face tracker based on motion prediction (Kalman-style velocity model)
and spatial/IoU bipartite matching (ByteTrack/SORT paradigm).
Maintains track continuity across skipped detection frames, handles temporary occlusions,
and assigns distinct temporary Track IDs separate from persistent Face IDs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger("katomaran.face_tracker")


def compute_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Compute Intersection-over-Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])

    inter_w = max(0.0, xb - xa)
    inter_h = max(0.0, yb - ya)
    inter_area = inter_w * inter_h

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])

    union_area = area_a + area_b - inter_area
    if union_area <= 0.0:
        return 0.0
    return float(inter_area / union_area)


@dataclass
class TrackedFace:
    """Internal representation of a face track across consecutive frames."""

    track_id: int
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float = 1.0
    face_id: Optional[str] = None  # Persistent visitor identity once recognized
    hits: int = 1                  # Total detection hits
    age: int = 1                   # Total frames since birth
    time_since_update: int = 0     # Consecutive frames without a detection association
    velocity: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])  # dx1, dy1, dx2, dy2

    def predict(self) -> List[float]:
        """Predict next bounding box position based on constant velocity motion model."""
        self.age += 1
        if self.time_since_update > 0:
            # Apply damped velocity prediction
            damping = 0.8
            p_bbox = [
                self.bbox[0] + self.velocity[0] * damping,
                self.bbox[1] + self.velocity[1] * damping,
                self.bbox[2] + self.velocity[2] * damping,
                self.bbox[3] + self.velocity[3] * damping,
            ]
            self.bbox = p_bbox
        self.time_since_update += 1
        return self.bbox

    def update(self, detection_bbox: Sequence[float], confidence: float) -> None:
        """Update track with fresh detection measurement."""
        alpha = 0.6  # Motion smoothing factor
        det = list(detection_bbox)
        # Update velocity vector
        new_vel = [det[i] - self.bbox[i] for i in range(4)]
        self.velocity = [
            (1.0 - alpha) * self.velocity[i] + alpha * new_vel[i]
            for i in range(4)
        ]
        self.bbox = det
        self.confidence = confidence
        self.hits += 1
        self.time_since_update = 0


class FaceTracker:
    """Multi-face association tracker that maintains track trajectories across video frames."""

    def __init__(
        self,
        iou_threshold: float = 0.25,
        max_missing_frames: int = 30,
        min_hits_to_confirm: int = 2,
        detection_skip_frames: int = 3,
    ):
        """Initialize FaceTracker.

        Args:
            iou_threshold: Minimum IoU overlap required to match a detection to an existing track.
            max_missing_frames: Maximum frames a track is kept without detection before deletion.
            min_hits_to_confirm: Minimum detection hits before track is considered confirmed.
            detection_skip_frames: How many frames are skipped between full detections.
        """
        self.iou_threshold = iou_threshold
        self.max_missing_frames = max_missing_frames
        self.min_hits_to_confirm = min_hits_to_confirm
        self.detection_skip_frames = detection_skip_frames

        self.next_track_id: int = 1
        self.tracks: Dict[int, TrackedFace] = {}

    def update(
        self,
        detections: Sequence[Tuple[Sequence[float], float]],
        is_detection_frame: bool = True,
    ) -> List[TrackedFace]:
        """Update tracker state for current frame.

        Args:
            detections: List of (bbox [x1, y1, x2, y2], confidence) for faces detected in this frame.
            is_detection_frame: If False, skips matching and advances tracks via motion prediction.

        Returns:
            List of active TrackedFace instances currently visible in this frame.
        """
        # Step 1: Predict new locations for all existing tracks
        for track in self.tracks.values():
            track.predict()

        if is_detection_frame:
            # Step 2: Associate detections with existing tracks using IoU matching
            matched_tracks, unmatched_dets, unmatched_tracks = self._associate_detections(detections)

            # Update matched tracks
            for track_id, det_idx in matched_tracks:
                det_box, det_conf = detections[det_idx]
                self.tracks[track_id].update(det_box, det_conf)

            # Create new tracks for unmatched detections
            for det_idx in unmatched_dets:
                det_box, det_conf = detections[det_idx]
                new_track = TrackedFace(
                    track_id=self.next_track_id,
                    bbox=list(det_box),
                    confidence=det_conf,
                    hits=1,
                    age=1,
                    time_since_update=0,
                )
                self.tracks[self.next_track_id] = new_track
                logger.debug(f"Created new video track #{self.next_track_id} at {det_box}.")
                self.next_track_id += 1

        # Step 3: Remove dead tracks exceeding max_missing_frames
        dead_ids = [
            tid
            for tid, t in self.tracks.items()
            if t.time_since_update > self.max_missing_frames
        ]
        for tid in dead_ids:
            logger.debug(f"Tracker pruned expired track #{tid} (missing {self.tracks[tid].time_since_update} frames).")
            del self.tracks[tid]

        # Step 4: Return confirmed and currently active tracks
        active_tracks: List[TrackedFace] = []
        max_active_gap = max(2, self.detection_skip_frames + 1)
        for track in self.tracks.values():
            # A track is active if updated recently (within detection interval) and confirmed
            if track.time_since_update <= max_active_gap and (track.hits >= self.min_hits_to_confirm or track.age <= max_active_gap):
                active_tracks.append(track)

        return active_tracks

    def _associate_detections(
        self, detections: Sequence[Tuple[Sequence[float], float]]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Perform greedy maximum-IoU bipartite matching between existing tracks and detections."""
        track_ids = list(self.tracks.keys())
        num_tracks = len(track_ids)
        num_dets = len(detections)

        if num_tracks == 0:
            return [], list(range(num_dets)), []

        if num_dets == 0:
            return [], [], track_ids

        # Compute IoU matrix (num_tracks x num_dets)
        iou_matrix = np.zeros((num_tracks, num_dets), dtype=np.float32)
        for t_idx, tid in enumerate(track_ids):
            track_box = self.tracks[tid].bbox
            for d_idx, (det_box, _) in enumerate(detections):
                iou_matrix[t_idx, d_idx] = compute_iou(track_box, det_box)

        # Match using greedy association starting with highest IoU
        matched_pairs: List[Tuple[int, int]] = []
        matched_t_indices = set()
        matched_d_indices = set()

        while True:
            if iou_matrix.size == 0:
                break
            max_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            max_val = iou_matrix[max_idx]

            if max_val < self.iou_threshold:
                break

            t_idx, d_idx = max_idx
            matched_pairs.append((track_ids[t_idx], d_idx))
            matched_t_indices.add(t_idx)
            matched_d_indices.add(d_idx)

            # Zero out row and column
            iou_matrix[t_idx, :] = -1.0
            iou_matrix[:, d_idx] = -1.0

        unmatched_dets = [d for d in range(num_dets) if d not in matched_d_indices]
        unmatched_tracks = [track_ids[t] for t in range(num_tracks) if t not in matched_t_indices]

        return matched_pairs, unmatched_dets, unmatched_tracks

    def associate_face_id(self, track_id: int, face_id: str) -> None:
        """Associate persistent Face ID with an active track."""
        if track_id in self.tracks:
            self.tracks[track_id].face_id = face_id
            logger.debug(f"Associated track #{track_id} with Face ID {face_id}.")
