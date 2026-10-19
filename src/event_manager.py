"""Track lifecycle state machine and event management for Katomaran Intelligent Face Tracker.

Enforces track lifecycle transitions:
    DETECTED -> IDENTIFIED -> ACTIVE -> EXIT_PENDING -> EXITED

Guarantees at most ONE ENTRY event and at most ONE EXIT event per confirmed track lifecycle,
prevents duplicate events, manages grace periods, and coordinates image persistence with SQLite.
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence

import numpy as np

from src.database import DatabaseManager, get_utc_iso_timestamp
from src.image_storage import ImageStorageManager

logger = logging.getLogger("katomaran.event_manager")


class LifecycleState(enum.Enum):
    DETECTED = "DETECTED"          # Track created, awaiting facial recognition
    IDENTIFIED = "IDENTIFIED"      # Face recognized/registered, ready for ENTRY event
    ACTIVE = "ACTIVE"              # ENTRY event committed, actively tracked in frame
    EXIT_PENDING = "EXIT_PENDING"  # Temporarily missing, within exit_grace_frames window
    EXITED = "EXITED"              # Confirmed gone, EXIT event committed, track closed


@dataclass
class TrackInfo:
    """Maintains state, coordinates, crops, and event status for a single tracked face."""

    track_id: int
    face_id: Optional[str] = None
    state: LifecycleState = LifecycleState.DETECTED
    last_bbox: Sequence[float] = field(default_factory=lambda: [0, 0, 0, 0])
    last_frame: Optional[np.ndarray] = None
    missing_frames: int = 0
    entry_event_id: Optional[int] = None
    exit_event_id: Optional[int] = None
    first_seen_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_seen_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    similarity_score: float = 0.0
    is_new_visitor: bool = False


class EventManager:
    """Coordinates lifecycle transitions, entry/exit emission, and database persistence."""

    def __init__(
        self,
        db_manager: DatabaseManager,
        image_storage: ImageStorageManager,
        exit_grace_frames: int = 30,
        end_of_stream_policy: str = "mark_active",
    ):
        self.db = db_manager
        self.storage = image_storage
        self.exit_grace_frames = max(1, exit_grace_frames)
        self.end_of_stream_policy = end_of_stream_policy

        # Active tracks keyed by temporary track_id
        self.tracks: Dict[int, TrackInfo] = {}

        # Session metrics (specific to this video / live run)
        self.session_entries: int = 0
        self.session_exits: int = 0
        self.session_unique_faces: set[str] = set()

    def update_track_detection(
        self,
        track_id: int,
        bbox: Sequence[float],
        frame: np.ndarray,
        face_id: Optional[str] = None,
        similarity_score: float = 0.0,
        is_new_visitor: bool = False,
    ) -> TrackInfo:
        """Update track with fresh observation in current frame.

        Args:
            track_id: Temporary tracker ID.
            bbox: [x1, y1, x2, y2] bounding box.
            frame: Current video frame.
            face_id: Assigned persistent Face ID if recognized or registered.
            similarity_score: Match confidence.
            is_new_visitor: True if freshly registered.

        Returns:
            Updated TrackInfo instance.
        """
        now = datetime.now(timezone.utc)
        if track_id not in self.tracks:
            # Transition to DETECTED
            track = TrackInfo(
                track_id=track_id,
                face_id=face_id,
                state=LifecycleState.DETECTED,
                last_bbox=bbox,
                last_frame=frame.copy() if frame is not None else None,
                missing_frames=0,
                first_seen_time=now,
                last_seen_time=now,
                similarity_score=similarity_score,
                is_new_visitor=is_new_visitor,
            )
            self.tracks[track_id] = track
            logger.info(f"Track #{track_id} initiated in state DETECTED.")
        else:
            track = self.tracks[track_id]
            track.last_bbox = bbox
            if frame is not None:
                track.last_frame = frame.copy()
            track.missing_frames = 0
            track.last_seen_time = now

            if face_id and not track.face_id:
                track.face_id = face_id
                track.similarity_score = similarity_score
                track.is_new_visitor = is_new_visitor

            # If track was in EXIT_PENDING, resume to ACTIVE
            if track.state == LifecycleState.EXIT_PENDING:
                track.state = LifecycleState.ACTIVE
                logger.info(
                    f"Track #{track_id} ({track.face_id}) resumed from EXIT_PENDING to ACTIVE."
                )

        # If track has a Face ID and hasn't committed ENTRY, trigger ENTRY
        if track.face_id and track.entry_event_id is None and track.state != LifecycleState.EXITED:
            self._commit_entry_event(track, frame, bbox)

        return track

    def _commit_entry_event(
        self,
        track: TrackInfo,
        frame: np.ndarray,
        bbox: Sequence[float],
    ) -> bool:
        """Idempotently trigger and commit an ENTRY event."""
        if track.entry_event_id is not None or not track.face_id:
            return False

        track.state = LifecycleState.IDENTIFIED
        ts_str = track.first_seen_time.isoformat()

        # 1. Save cropped face image
        img_path = self.storage.save_face_image(
            frame=frame,
            bbox=bbox,
            face_id=track.face_id,
            event_type="ENTRY",
            timestamp=track.first_seen_time,
        )

        if not img_path:
            logger.error(
                f"Failed to save ENTRY crop for {track.face_id} (track #{track.track_id})."
            )
            # Use placeholder path to avoid blocking database consistency
            img_path = f"logs/entries/{track.face_id}_unavailable.jpg"

        # 2. Persist event in database
        event_id = self.db.record_event(
            face_id=track.face_id,
            event_type="ENTRY",
            image_path=img_path,
            timestamp=ts_str,
            track_id=track.track_id,
        )

        if event_id is not None:
            track.entry_event_id = event_id
            track.state = LifecycleState.ACTIVE
            self.session_entries += 1
            if track.face_id:
                self.session_unique_faces.add(track.face_id)
            logger.info(
                f"[EVENT: ENTRY] Confirmed for {track.face_id} on track #{track.track_id} "
                f"(Event #{event_id}, Image: {img_path})"
            )
            return True
        else:
            logger.error(f"Failed to record ENTRY database record for {track.face_id}.")
            return False

    def process_missing_tracks(self, active_track_ids_in_frame: Sequence[int]) -> List[TrackInfo]:
        """Check all active tracks against tracks observed in the current frame.
        Handles grace period counting and confirms exits when threshold exceeded.

        Args:
            active_track_ids_in_frame: Set or list of track IDs detected/tracked in this frame.

        Returns:
            List of TrackInfo instances that transitioned to EXITED in this step.
        """
        active_set = set(active_track_ids_in_frame)
        exited_tracks: List[TrackInfo] = []

        # Iterate over copy of keys
        for track_id, track in list(self.tracks.items()):
            if track.state == LifecycleState.EXITED:
                continue

            if track_id not in active_set:
                track.missing_frames += 1

                if track.missing_frames < self.exit_grace_frames:
                    if track.state == LifecycleState.ACTIVE:
                        track.state = LifecycleState.EXIT_PENDING
                        logger.debug(
                            f"Track #{track_id} missing ({track.missing_frames}/{self.exit_grace_frames} frames). "
                            "State: EXIT_PENDING."
                        )
                else:
                    # Grace period expired -> Confirm EXIT
                    self._commit_exit_event(track)
                    exited_tracks.append(track)

        return exited_tracks

    def _commit_exit_event(self, track: TrackInfo) -> bool:
        """Idempotently trigger and commit an EXIT event."""
        if track.exit_event_id is not None:
            return False

        if not track.face_id:
            # Track was never identified before disappearing; close without event
            track.state = LifecycleState.EXITED
            logger.info(f"Unidentified track #{track.track_id} expired and closed without events.")
            return False

        ts_str = track.last_seen_time.isoformat()

        # Crop face from last known frame if available
        img_path = None
        if track.last_frame is not None:
            img_path = self.storage.save_face_image(
                frame=track.last_frame,
                bbox=track.last_bbox,
                face_id=track.face_id,
                event_type="EXIT",
                timestamp=track.last_seen_time,
            )

        if not img_path:
            logger.warning(
                f"Could not save EXIT crop for {track.face_id} (track #{track.track_id}). Using fallback."
            )
            img_path = f"logs/exits/{track.face_id}_exit_crop.jpg"

        # Record in database
        event_id = self.db.record_event(
            face_id=track.face_id,
            event_type="EXIT",
            image_path=img_path,
            timestamp=ts_str,
            track_id=track.track_id,
        )

        if event_id is not None:
            track.exit_event_id = event_id
            track.state = LifecycleState.EXITED
            self.session_exits += 1
            logger.info(
                f"[EVENT: EXIT] Confirmed for {track.face_id} on track #{track.track_id} "
                f"after {track.missing_frames} missing frames. (Event #{event_id})"
            )
            return True
        else:
            logger.error(f"Failed to record EXIT database record for {track.face_id}.")
            return False

    @property
    def session_entry_count(self) -> int:
        """Count of ENTRY events recorded in this run."""
        return self.session_entries

    @property
    def session_exit_count(self) -> int:
        """Count of EXIT events recorded in this run."""
        return self.session_exits

    @property
    def session_visitor_count(self) -> int:
        """Count of distinct recognized visitors who entered during this run."""
        return len(self.session_unique_faces)

    def handle_end_of_stream(self) -> List[TrackInfo]:
        """Apply end-of-stream policy for remaining active tracks.

        - 'flush_exits': Force confirm EXIT for all open tracks.
        - 'mark_active': Keep active tracks open and document status in database.
        """
        closed_tracks: List[TrackInfo] = []
        logger.info(f"Applying end-of-stream policy: '{self.end_of_stream_policy}'.")

        for track_id, track in self.tracks.items():
            if track.state in (LifecycleState.ACTIVE, LifecycleState.EXIT_PENDING):
                if self.end_of_stream_policy == "flush_exits":
                    self._commit_exit_event(track)
                    closed_tracks.append(track)
                else:
                    # mark_active: Record current state in database without fabricating exit
                    if track.face_id:
                        self.db.update_visitor_activity(
                            face_id=track.face_id,
                            timestamp=track.last_seen_time.isoformat(),
                            status="active_at_stream_end",
                        )
                        logger.info(
                            f"Track #{track_id} ({track.face_id}) left in status 'active_at_stream_end'."
                        )

        return closed_tracks

    def get_active_tracks_summary(self) -> List[Dict[str, Any]]:
        """Return structured summary of currently tracked identities."""
        summary = []
        for tid, t in self.tracks.items():
            if t.state != LifecycleState.EXITED:
                summary.append({
                    "track_id": tid,
                    "face_id": t.face_id,
                    "state": t.state.value,
                    "missing_frames": t.missing_frames,
                    "first_seen": t.first_seen_time.isoformat(),
                    "last_seen": t.last_seen_time.isoformat(),
                })
        return summary
