"""Configuration module for Katomaran Intelligent Face Tracker.

Loads, validates, and provides access to application settings with path resolution
relative to the project root.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    # This file is located in <project_root>/src/config.py
    return Path(__file__).resolve().parent.parent


@dataclass
class AppConfig:
    """Application configuration schema with validation and relative path resolution."""

    # Video input settings
    video_source: str = "videos/sample_video.mp4"
    input_type: str = "auto"  # 'auto', 'video', 'rtsp', 'webcam'
    frame_resize_width: int = 960

    # Detection and tracking intervals
    # detection_skip_frames: detection is performed every (detection_skip_frames + 1) frames.
    # E.g., if set to 3, frames 0, 4, 8... undergo full YOLO detection; in-between frames use tracker prediction.
    detection_skip_frames: int = 3
    exit_grace_frames: int = 30  # consecutive missing frames before confirming track exit

    # Face thresholds
    face_confidence_threshold: float = 0.40
    min_face_size: int = 20  # Minimum width/height in pixels
    recognition_threshold: float = 0.45  # Cosine similarity threshold for ArcFace embeddings

    # Model paths and parameters
    yolo_model_path: str = "models/face_detector.pt"
    recognition_model_name: str = "buffalo_l"

    # Storage and persistence paths (relative to project root)
    database_path: str = "database/visitors.db"
    entry_image_dir: str = "logs/entries"
    exit_image_dir: str = "logs/exits"
    event_log_path: str = "logs/events.log"

    # Runtime behavior
    display_video: bool = True
    save_detection_debug_video: bool = False
    log_level: str = "INFO"

    # RTSP specific settings
    rtsp_reconnect_delay_sec: float = 2.0
    rtsp_max_reconnect_attempts: int = 5

    # End of stream policy: 'mark_active' (leave active tracks open) or 'flush_exits' (emit exit for remaining)
    end_of_stream_policy: str = "mark_active"

    # Internal project root reference
    project_root: Path = field(default_factory=get_project_root)

    def validate(self) -> None:
        """Validate configuration settings and raise ValueError on invalid values."""
        if not self.video_source:
            raise ValueError("Configuration error: 'video_source' must not be empty.")

        valid_input_types = {"auto", "video", "rtsp", "webcam"}
        if self.input_type.lower() not in valid_input_types:
            raise ValueError(
                f"Configuration error: Invalid 'input_type' '{self.input_type}'. "
                f"Supported types are: {', '.join(sorted(valid_input_types))}."
            )

        if self.detection_skip_frames < 0:
            raise ValueError(
                f"Configuration error: 'detection_skip_frames' must be >= 0 (got {self.detection_skip_frames})."
            )

        if self.exit_grace_frames < 1:
            raise ValueError(
                f"Configuration error: 'exit_grace_frames' must be >= 1 (got {self.exit_grace_frames})."
            )

        if not (0.0 <= self.recognition_threshold <= 1.0):
            raise ValueError(
                f"Configuration error: 'recognition_threshold' must be between 0.0 and 1.0 (got {self.recognition_threshold})."
            )

        if not (0.0 <= self.face_confidence_threshold <= 1.0):
            raise ValueError(
                f"Configuration error: 'face_confidence_threshold' must be between 0.0 and 1.0 (got {self.face_confidence_threshold})."
            )

        if self.min_face_size < 10:
            raise ValueError(
                f"Configuration error: 'min_face_size' must be at least 10 pixels (got {self.min_face_size})."
            )

        if self.frame_resize_width < 100:
            raise ValueError(
                f"Configuration error: 'frame_resize_width' must be at least 100 pixels (got {self.frame_resize_width})."
            )

        valid_log_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.log_level.upper() not in valid_log_levels:
            raise ValueError(
                f"Configuration error: Invalid 'log_level' '{self.log_level}'. "
                f"Supported levels are: {', '.join(sorted(valid_log_levels))}."
            )

        valid_stream_policies = {"mark_active", "flush_exits"}
        if self.end_of_stream_policy.lower() not in valid_stream_policies:
            raise ValueError(
                f"Configuration error: Invalid 'end_of_stream_policy' '{self.end_of_stream_policy}'. "
                f"Supported policies are: {', '.join(sorted(valid_stream_policies))}."
            )

    def resolve_path(self, rel_or_abs_path: str | Path) -> Path:
        """Resolve a path relative to the project root, unless already absolute."""
        p = Path(rel_or_abs_path)
        if p.is_absolute():
            return p
        return (self.project_root / p).resolve()

    def get_resolved_database_path(self) -> Path:
        """Get absolute path to SQLite database."""
        return self.resolve_path(self.database_path)

    def get_resolved_entry_dir(self) -> Path:
        """Get absolute path to entry cropped images directory."""
        return self.resolve_path(self.entry_image_dir)

    def get_resolved_exit_dir(self) -> Path:
        """Get absolute path to exit cropped images directory."""
        return self.resolve_path(self.exit_image_dir)

    def get_resolved_log_path(self) -> Path:
        """Get absolute path to event log file."""
        return self.resolve_path(self.event_log_path)

    def get_resolved_yolo_path(self) -> Path:
        """Get absolute path to YOLO model file."""
        return self.resolve_path(self.yolo_model_path)

    def get_resolved_video_source(self) -> str:
        """Resolve video source path if it refers to a local file, or return URL/index as-is."""
        src = self.video_source.strip()
        # Check if it's an RTSP / HTTP URL or integer webcam index
        if src.startswith(("rtsp://", "rtsps://", "http://", "https://")):
            return src
        if src.isdigit():
            return src
        # Local file path
        return str(self.resolve_path(src))

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary (excluding non-serializable fields)."""
        data = asdict(self)
        data.pop("project_root", None)
        return data


def load_config(
    config_path: Optional[str | Path] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> AppConfig:
    """Load configuration from JSON file with optional runtime overrides.

    Args:
        config_path: Path to config.json (defaults to config.json at project root).
        overrides: Dictionary of settings to override (e.g., from command line).

    Returns:
        Validated AppConfig instance.
    """
    root = get_project_root()
    target_path = Path(config_path) if config_path else (root / "config.json")
    if not target_path.is_absolute():
        target_path = (root / target_path).resolve()

    data: Dict[str, Any] = {}
    if target_path.exists():
        try:
            with open(target_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to read configuration file at '{target_path}': {e}") from e
    else:
        # If config file doesn't exist, we fall back to defaults
        pass

    if overrides:
        for k, v in overrides.items():
            if v is not None:
                data[k] = v

    # Create dataclass instance filtering out unknown keys
    valid_fields = {f for f in AppConfig.__dataclass_fields__ if f != "project_root"}
    filtered_data = {k: v for k, v in data.items() if k in valid_fields}

    config = AppConfig(project_root=root, **filtered_data)
    config.validate()
    return config
