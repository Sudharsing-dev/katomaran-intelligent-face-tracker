"""Unit tests for configuration loading, validation, and relative path resolution."""

import json
import pytest
from pathlib import Path

from src.config import AppConfig, load_config, get_project_root


def test_default_config_validity():
    config = AppConfig()
    config.validate()
    assert config.detection_skip_frames == 3
    assert config.exit_grace_frames == 30
    assert config.recognition_threshold == 0.45
    assert config.input_type == "auto"


def test_relative_path_resolution(tmp_path):
    config = AppConfig(project_root=tmp_path, database_path="database/visitors.db")
    resolved = config.get_resolved_database_path()
    assert resolved == (tmp_path / "database" / "visitors.db").resolve()


def test_rtsp_url_not_altered():
    config = AppConfig(video_source="rtsp://192.168.1.100:554/live")
    assert config.get_resolved_video_source() == "rtsp://192.168.1.100:554/live"


def test_invalid_skip_frames():
    config = AppConfig(detection_skip_frames=-1)
    with pytest.raises(ValueError, match="detection_skip_frames"):
        config.validate()


def test_invalid_exit_grace():
    config = AppConfig(exit_grace_frames=0)
    with pytest.raises(ValueError, match="exit_grace_frames"):
        config.validate()


def test_invalid_recognition_threshold():
    config = AppConfig(recognition_threshold=1.5)
    with pytest.raises(ValueError, match="recognition_threshold"):
        config.validate()


def test_invalid_input_type():
    config = AppConfig(input_type="invalid_stream")
    with pytest.raises(ValueError, match="input_type"):
        config.validate()


def test_load_config_with_overrides(tmp_path):
    cfg_file = tmp_path / "custom_config.json"
    data = {
        "video_source": "custom.mp4",
        "detection_skip_frames": 5,
        "recognition_threshold": 0.50,
    }
    cfg_file.write_text(json.dumps(data), encoding="utf-8")

    loaded = load_config(config_path=cfg_file, overrides={"recognition_threshold": 0.60})
    assert loaded.video_source == "custom.mp4"
    assert loaded.detection_skip_frames == 5
    assert loaded.recognition_threshold == 0.60  # Overridden
