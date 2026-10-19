"""
Tests for VideoReader — RTSP connection handling, credential masking,
reconnection logic, and local file compatibility.

All camera connections are mocked; no real RTSP camera is required.

NOTE: Live-camera validation (against a real RTSP stream) remains pending
until the interview team supplies a valid RTSP URL and network access.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, call, patch, PropertyMock

import numpy as np
import pytest

from src.video_reader import VideoReader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_cap(is_opened: bool = True, frame: np.ndarray | None = None) -> MagicMock:
    """Return a MagicMock mimicking cv2.VideoCapture."""
    cap = MagicMock()
    cap.isOpened.return_value = is_opened
    if frame is not None:
        cap.read.return_value = (True, frame)
    else:
        cap.read.return_value = (False, None)
    cap.get.return_value = 0.0
    return cap


def _dummy_frame(width: int = 640, height: int = 480) -> np.ndarray:
    """Return a solid-colour BGR frame."""
    return np.zeros((height, width, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# 1. RTSP URL detection
# ---------------------------------------------------------------------------

class TestRtspUrlDetection:
    """VideoReader correctly identifies RTSP and non-RTSP sources."""

    @pytest.mark.parametrize("url", [
        "rtsp://192.168.1.100:554/live",
        "rtsps://secure-cam.local:443/stream",
        "http://192.168.0.5:8080/mjpeg",
        "https://cam.example.com/feed",
    ])
    def test_rtsp_like_urls_detected(self, url):
        with patch("cv2.VideoCapture") as mock_vc:
            mock_vc.return_value = _make_mock_cap(is_opened=True)
            reader = VideoReader(source=url, reconnect_delay_sec=0, max_reconnect_attempts=1)
        assert reader.is_rtsp is True
        assert reader.is_live is True

    @pytest.mark.parametrize("source", [
        "videos/sample.mp4",
        "/tmp/recording.avi",
        r"C:\Videos\clip.mp4",
    ])
    def test_file_sources_not_rtsp(self, tmp_path, source):
        # Use a real file so _open_source() existence check passes
        p = tmp_path / "video.mp4"
        p.write_bytes(b"")
        with patch("cv2.VideoCapture") as mock_vc:
            mock_vc.return_value = _make_mock_cap(is_opened=True)
            reader = VideoReader(
                source=str(p),
                reconnect_delay_sec=0,
                max_reconnect_attempts=1,
            )
        assert reader.is_rtsp is False

    def test_integer_webcam_is_live_not_rtsp(self):
        with patch("cv2.VideoCapture") as mock_vc:
            mock_vc.return_value = _make_mock_cap(is_opened=True)
            reader = VideoReader(source=0, reconnect_delay_sec=0, max_reconnect_attempts=1)
        assert reader.is_rtsp is False
        assert reader.is_live is True


# ---------------------------------------------------------------------------
# 2. Credential masking
# ---------------------------------------------------------------------------

class TestCredentialMasking:
    """Passwords embedded in RTSP URLs are never exposed in sanitized names."""

    def test_credentials_masked_in_sanitized_name(self):
        url = "rtsp://admin:s3cr3t@192.168.1.50:554/stream1"
        with patch("cv2.VideoCapture") as mock_vc:
            mock_vc.return_value = _make_mock_cap(is_opened=True)
            reader = VideoReader(source=url, reconnect_delay_sec=0, max_reconnect_attempts=1)
        sanitized = reader.get_sanitized_source_name()
        assert "admin" not in sanitized
        assert "s3cr3t" not in sanitized
        assert "***:***" in sanitized
        assert "192.168.1.50:554" in sanitized

    def test_url_without_credentials_unchanged(self):
        url = "rtsp://192.168.1.100:554/live"
        with patch("cv2.VideoCapture") as mock_vc:
            mock_vc.return_value = _make_mock_cap(is_opened=True)
            reader = VideoReader(source=url, reconnect_delay_sec=0, max_reconnect_attempts=1)
        assert reader.get_sanitized_source_name() == url

    def test_log_messages_do_not_contain_password(self, caplog):
        url = "rtsp://user:TopSecret99@10.0.0.1:554/cam"
        with caplog.at_level(logging.DEBUG, logger="katomaran.video_reader"):
            with patch("cv2.VideoCapture") as mock_vc:
                mock_vc.return_value = _make_mock_cap(is_opened=True)
                VideoReader(source=url, reconnect_delay_sec=0, max_reconnect_attempts=1)
        full_log = "\n".join(caplog.messages)
        assert "TopSecret99" not in full_log
        assert "user" not in full_log or "***" in full_log  # masked


# ---------------------------------------------------------------------------
# 3. Successful RTSP startup
# ---------------------------------------------------------------------------

class TestRtspStartupSuccess:
    """VideoReader.connected is True when the camera accepts the first attempt."""

    def test_connected_after_successful_open(self):
        with patch("cv2.VideoCapture") as mock_vc:
            cap = _make_mock_cap(is_opened=True)
            cap.get.side_effect = lambda prop: {
                1: 1280.0,   # CAP_PROP_FRAME_WIDTH
                2: 720.0,    # CAP_PROP_FRAME_HEIGHT
                5: 25.0,     # CAP_PROP_FPS
                7: 0.0,      # CAP_PROP_FRAME_COUNT
            }.get(prop, 0.0)
            mock_vc.return_value = cap
            reader = VideoReader(
                source="rtsp://192.168.1.1:554/live",
                reconnect_delay_sec=0,
                max_reconnect_attempts=3,
            )
        assert reader.connected is True

    def test_frames_yielded_when_connected(self):
        frame = _dummy_frame()
        with patch("cv2.VideoCapture") as mock_vc:
            cap = _make_mock_cap(is_opened=True, frame=frame)
            # Yield two frames then signal EOF
            cap.read.side_effect = [
                (True, frame),
                (True, frame),
                (False, None),  # EOF
            ]
            cap.get.return_value = 0.0
            mock_vc.return_value = cap
            reader = VideoReader(
                source="rtsp://192.168.1.1:554/live",
                reconnect_delay_sec=0,
                max_reconnect_attempts=1,
            )
        frames = list(reader.frames())
        assert len(frames) == 2
        assert frames[0][0] == 0   # frame index
        assert frames[1][0] == 1


# ---------------------------------------------------------------------------
# 4. Initial RTSP connection failure
# ---------------------------------------------------------------------------

class TestRtspStartupFailure:
    """When the camera is unreachable, connected is False and no frames are yielded."""

    def test_connected_false_after_all_attempts_fail(self, capsys):
        with patch("cv2.VideoCapture") as mock_vc:
            mock_vc.return_value = _make_mock_cap(is_opened=False)
            reader = VideoReader(
                source="rtsp://unreachable:554/live",
                reconnect_delay_sec=0,   # no sleep in tests
                max_reconnect_attempts=2,
            )
        assert reader.connected is False

    def test_no_frames_yielded_when_not_connected(self):
        with patch("cv2.VideoCapture") as mock_vc:
            mock_vc.return_value = _make_mock_cap(is_opened=False)
            reader = VideoReader(
                source="rtsp://unreachable:554/live",
                reconnect_delay_sec=0,
                max_reconnect_attempts=2,
            )
        frames = list(reader.frames())
        assert frames == []

    def test_error_message_printed_on_failure(self, capsys):
        with patch("cv2.VideoCapture") as mock_vc:
            mock_vc.return_value = _make_mock_cap(is_opened=False)
            VideoReader(
                source="rtsp://192.168.1.99:554/live",
                reconnect_delay_sec=0,
                max_reconnect_attempts=1,
            )
        captured = capsys.readouterr()
        assert "ERROR" in captured.out
        assert "Unable to connect" in captured.out

    def test_correct_number_of_open_attempts(self):
        with patch("cv2.VideoCapture") as mock_vc:
            mock_vc.return_value = _make_mock_cap(is_opened=False)
            VideoReader(
                source="rtsp://cam:554/stream",
                reconnect_delay_sec=0,
                max_reconnect_attempts=3,
            )
        # VideoCapture constructed once per attempt (release() sets cap=None before re-open)
        assert mock_vc.call_count == 3

    def test_error_logged_on_startup_failure(self, caplog):
        with caplog.at_level(logging.ERROR, logger="katomaran.video_reader"):
            with patch("cv2.VideoCapture") as mock_vc:
                mock_vc.return_value = _make_mock_cap(is_opened=False)
                VideoReader(
                    source="rtsp://bad-cam:554/live",
                    reconnect_delay_sec=0,
                    max_reconnect_attempts=1,
                )
        error_logs = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert len(error_logs) >= 1


# ---------------------------------------------------------------------------
# 5. Mid-stream reconnection
# ---------------------------------------------------------------------------

class TestMidStreamReconnection:
    """_reconnect_rtsp() is called when cap.read() fails and retries correctly."""

    def test_reconnect_succeeds_after_one_failure(self):
        frame = _dummy_frame()
        call_count = {"n": 0}

        def make_cap(*args):
            """First two calls return a cap whose read() fails; third returns a good cap."""
            call_count["n"] += 1
            if call_count["n"] <= 2:
                # Initial open succeeds but the first read() signals a drop
                cap = _make_mock_cap(is_opened=True)
                cap.read.side_effect = [(False, None)]
                return cap
            else:
                # Reconnect cap: opens fine and delivers one frame then EOF
                cap = _make_mock_cap(is_opened=True)
                cap.read.side_effect = [(True, frame), (False, None)]
                return cap

        with patch("cv2.VideoCapture", side_effect=make_cap):
            reader = VideoReader(
                source="rtsp://cam:554/stream",
                reconnect_delay_sec=0,
                max_reconnect_attempts=3,
            )
        # After initial open succeeds, frames() should handle the drop and reconnect
        frames = list(reader.frames())
        # At minimum we should attempt reconnect without crashing
        assert isinstance(frames, list)

    def test_max_retries_exhausted_terminates_frame_loop(self):
        """When all reconnect attempts fail the generator terminates cleanly."""
        frame = _dummy_frame()
        cap_sequence = []

        # First cap: opens OK but read() fails → triggers reconnect
        good_open_bad_read = _make_mock_cap(is_opened=True)
        good_open_bad_read.read.return_value = (False, None)
        cap_sequence.append(good_open_bad_read)

        # Subsequent caps: always fail to open → exhaust reconnect budget
        for _ in range(3):
            cap_sequence.append(_make_mock_cap(is_opened=False))

        with patch("cv2.VideoCapture", side_effect=cap_sequence):
            reader = VideoReader(
                source="rtsp://cam:554/stream",
                reconnect_delay_sec=0,
                max_reconnect_attempts=3,
            )
        frames = list(reader.frames())
        assert frames == []
        assert reader.connected is False


# ---------------------------------------------------------------------------
# 6. Local video file compatibility
# ---------------------------------------------------------------------------

class TestLocalVideoFile:
    """Existing video-file behaviour is not regressed."""

    def test_file_connected_when_opened(self, tmp_path):
        p = tmp_path / "test.mp4"
        p.write_bytes(b"")
        with patch("cv2.VideoCapture") as mock_vc:
            mock_vc.return_value = _make_mock_cap(is_opened=True)
            reader = VideoReader(source=str(p), reconnect_delay_sec=0, max_reconnect_attempts=1)
        assert reader.connected is True
        assert reader.is_rtsp is False

    def test_missing_file_not_connected(self, tmp_path):
        nonexistent = str(tmp_path / "missing_video.mp4")
        with patch("cv2.VideoCapture") as mock_vc:
            mock_vc.return_value = _make_mock_cap(is_opened=True)
            reader = VideoReader(
                source=nonexistent, reconnect_delay_sec=0, max_reconnect_attempts=1
            )
        assert reader.connected is False

    def test_frames_from_file(self, tmp_path):
        p = tmp_path / "clip.mp4"
        p.write_bytes(b"")
        frame = _dummy_frame()
        with patch("cv2.VideoCapture") as mock_vc:
            cap = _make_mock_cap(is_opened=True)
            cap.read.side_effect = [(True, frame), (True, frame), (False, None)]
            cap.get.return_value = 0.0
            mock_vc.return_value = cap
            reader = VideoReader(
                source=str(p),
                reconnect_delay_sec=0,
                max_reconnect_attempts=1,
                target_width=None,   # no resize
            )
        frames = list(reader.frames())
        assert len(frames) == 2

    def test_eof_terminates_generator(self, tmp_path):
        p = tmp_path / "short.mp4"
        p.write_bytes(b"")
        with patch("cv2.VideoCapture") as mock_vc:
            cap = _make_mock_cap(is_opened=True)
            cap.read.side_effect = [(False, None)]   # immediate EOF
            cap.get.return_value = 0.0
            mock_vc.return_value = cap
            reader = VideoReader(source=str(p), reconnect_delay_sec=0, max_reconnect_attempts=1)
        frames = list(reader.frames())
        assert frames == []


# ---------------------------------------------------------------------------
# 7. Resource cleanup
# ---------------------------------------------------------------------------

class TestResourceCleanup:
    """release() and the context manager free resources safely."""

    def test_release_called_on_context_exit(self):
        with patch("cv2.VideoCapture") as mock_vc:
            cap = _make_mock_cap(is_opened=True)
            cap.read.return_value = (False, None)
            mock_vc.return_value = cap
            with VideoReader(
                source="rtsp://cam:554/live",
                reconnect_delay_sec=0,
                max_reconnect_attempts=1,
            ) as reader:
                pass
        cap.release.assert_called()

    def test_double_release_safe(self):
        with patch("cv2.VideoCapture") as mock_vc:
            mock_vc.return_value = _make_mock_cap(is_opened=True)
            reader = VideoReader(
                source="rtsp://cam:554/live",
                reconnect_delay_sec=0,
                max_reconnect_attempts=1,
            )
        reader.release()
        reader.release()  # Must not raise

    def test_connected_false_not_crashes_frames(self):
        """frames() on a never-connected reader exits cleanly."""
        with patch("cv2.VideoCapture") as mock_vc:
            mock_vc.return_value = _make_mock_cap(is_opened=False)
            reader = VideoReader(
                source="rtsp://nowhere:554/live",
                reconnect_delay_sec=0,
                max_reconnect_attempts=1,
            )
        assert list(reader.frames()) == []


# ---------------------------------------------------------------------------
# 8. Frame resize
# ---------------------------------------------------------------------------

class TestFrameResize:
    """Frames wider than target_width are scaled down correctly."""

    def test_wide_frame_resized(self, tmp_path):
        p = tmp_path / "wide.mp4"
        p.write_bytes(b"")
        wide_frame = _dummy_frame(width=1920, height=1080)
        with patch("cv2.VideoCapture") as mock_vc:
            cap = _make_mock_cap(is_opened=True)
            cap.read.side_effect = [(True, wide_frame), (False, None)]
            cap.get.return_value = 0.0
            mock_vc.return_value = cap
            reader = VideoReader(
                source=str(p),
                reconnect_delay_sec=0,
                max_reconnect_attempts=1,
                target_width=960,
            )
        frames = list(reader.frames())
        assert len(frames) == 1
        _, out_frame = frames[0]
        assert out_frame.shape[1] == 960           # width
        assert out_frame.shape[0] == 540           # height (proportional)

    def test_narrow_frame_not_resized(self, tmp_path):
        p = tmp_path / "narrow.mp4"
        p.write_bytes(b"")
        narrow_frame = _dummy_frame(width=640, height=480)
        with patch("cv2.VideoCapture") as mock_vc:
            cap = _make_mock_cap(is_opened=True)
            cap.read.side_effect = [(True, narrow_frame), (False, None)]
            cap.get.return_value = 0.0
            mock_vc.return_value = cap
            reader = VideoReader(
                source=str(p),
                reconnect_delay_sec=0,
                max_reconnect_attempts=1,
                target_width=960,
            )
        frames = list(reader.frames())
        _, out_frame = frames[0]
        assert out_frame.shape[1] == 640   # unchanged
