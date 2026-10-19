"""Tests for the upload_processor module.

All camera, model, and filesystem interactions are mocked so these tests
run without installed AI models, a real camera, or a real database.
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from src.upload_processor import (
    ALLOWED_IMAGE_EXTS,
    ALLOWED_VIDEO_EXTS,
    _parse_pipeline_summary,
    draw_detections,
    is_safe_upload_filename,
    process_image_upload,
    safe_unique_video_filename,
    save_uploaded_video,
    validate_and_decode_image,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_jpeg_bytes(width: int = 80, height: int = 60) -> bytes:
    """Return valid JPEG bytes for a small blank image."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok, "cv2.imencode unexpectedly failed in test helper"
    return buf.tobytes()


def _dummy_frame(w: int = 640, h: int = 480) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# 1. validate_and_decode_image
# ---------------------------------------------------------------------------

class TestValidateAndDecodeImage:
    def test_valid_jpeg_returns_ndarray(self):
        data = _make_jpeg_bytes()
        frame = validate_and_decode_image(data)
        assert frame is not None
        assert isinstance(frame, np.ndarray)
        assert frame.ndim == 3

    def test_empty_bytes_returns_none(self):
        assert validate_and_decode_image(b"") is None

    def test_none_bytes_returns_none(self):
        # Passing None is a programming error but should be handled gracefully
        assert validate_and_decode_image(None) is None  # type: ignore[arg-type]

    def test_garbage_bytes_returns_none(self):
        assert validate_and_decode_image(b"\x00\xff\xab\xcd" * 10) is None

    def test_large_image_is_down_scaled(self):
        """Images wider than MAX_IMAGE_PIXELS must be resized."""
        big_img = np.zeros((5000, 5000, 3), dtype=np.uint8)
        ok, buf = cv2.imencode(".png", big_img)
        assert ok
        frame = validate_and_decode_image(buf.tobytes())
        assert frame is not None
        # Longest dimension must not exceed the limit
        from src.upload_processor import MAX_IMAGE_PIXELS
        assert max(frame.shape[:2]) <= MAX_IMAGE_PIXELS

    def test_small_image_not_resized(self):
        data = _make_jpeg_bytes(100, 80)
        frame = validate_and_decode_image(data)
        assert frame is not None
        assert frame.shape[1] == 100  # width unchanged


# ---------------------------------------------------------------------------
# 2. is_safe_upload_filename
# ---------------------------------------------------------------------------

class TestIsSafeUploadFilename:
    @pytest.mark.parametrize("name", ["photo.jpg", "face.jpeg", "scan.png",
                                       "clip.mp4", "recording.avi", "demo.mov"])
    def test_allowed_extensions_are_safe(self, name):
        assert is_safe_upload_filename(name) is True

    @pytest.mark.parametrize("name", [
        "../config.json",
        "..\\src\\config.py",
        "subdir/video.mp4",
        "a/b/c.jpg",
    ])
    def test_path_traversal_rejected(self, name):
        assert is_safe_upload_filename(name) is False

    @pytest.mark.parametrize("name", ["script.py", "attack.exe", "db.db", "no_ext"])
    def test_disallowed_extensions_rejected(self, name):
        assert is_safe_upload_filename(name) is False


# ---------------------------------------------------------------------------
# 3. safe_unique_video_filename
# ---------------------------------------------------------------------------

class TestSafeUniqueVideoFilename:
    def test_returns_string(self):
        result = safe_unique_video_filename("my video.mp4")
        assert isinstance(result, str)

    def test_preserves_extension_lowercase(self):
        assert safe_unique_video_filename("clip.MP4").endswith(".mp4")
        assert safe_unique_video_filename("recording.AVI").endswith(".avi")

    def test_has_uuid_prefix(self):
        name = safe_unique_video_filename("test.mp4")
        # First 10 chars should be hex UUID fragment
        parts = name.split("_", 1)
        assert len(parts[0]) == 10
        assert all(c in "0123456789abcdef" for c in parts[0])

    def test_special_chars_sanitised(self):
        name = safe_unique_video_filename("bad name!@#$.mp4")
        # After the UUID prefix, no special chars except _ and -
        stem = Path(name).stem[11:]  # skip uuid prefix + underscore
        assert re.match(r"^[\w\-]*$", stem), f"Unsafe stem: {stem!r}"

    def test_two_calls_produce_different_names(self):
        a = safe_unique_video_filename("same.mp4")
        b = safe_unique_video_filename("same.mp4")
        assert a != b


# ---------------------------------------------------------------------------
# 4. save_uploaded_video
# ---------------------------------------------------------------------------

class TestSaveUploadedVideo:
    def test_file_created_with_correct_content(self, tmp_path):
        data = b"fakevideobytes"
        dest = save_uploaded_video(data, "clip.mp4", tmp_path)
        assert dest.exists()
        assert dest.read_bytes() == data

    def test_creates_missing_directory(self, tmp_path):
        upload_dir = tmp_path / "videos" / "uploads"
        assert not upload_dir.exists()
        save_uploaded_video(b"data", "v.mp4", upload_dir)
        assert upload_dir.exists()

    def test_unique_filenames_prevent_overwrite(self, tmp_path):
        dest1 = save_uploaded_video(b"aaa", "clip.mp4", tmp_path)
        dest2 = save_uploaded_video(b"bbb", "clip.mp4", tmp_path)
        assert dest1 != dest2
        assert dest1.exists() and dest2.exists()

    def test_filename_is_safe(self, tmp_path):
        dest = save_uploaded_video(b"x", "my bad file!.mp4", tmp_path)
        # Filename should not contain spaces or ! characters
        assert " " not in dest.name
        assert "!" not in dest.name


# ---------------------------------------------------------------------------
# 5. draw_detections
# ---------------------------------------------------------------------------

class TestDrawDetections:
    def test_no_detections_returns_same_shape(self):
        frame = _dummy_frame()
        result = draw_detections(frame, [])
        assert result.shape == frame.shape

    def test_does_not_mutate_input_frame(self):
        frame = _dummy_frame()
        original = frame.copy()
        draw_detections(frame, [([10, 10, 100, 100], 0.9)])
        np.testing.assert_array_equal(frame, original)

    def test_with_recognized_face(self):
        frame = _dummy_frame(200, 200)
        result = draw_detections(frame, [([20, 20, 80, 80], 0.85)], face_ids=["FACE_001"])
        assert result.shape == frame.shape
        # The annotated image should differ from the blank input
        assert not np.array_equal(result, frame)

    def test_with_unknown_face(self):
        frame = _dummy_frame(200, 200)
        result = draw_detections(frame, [([20, 20, 80, 80], 0.75)], face_ids=[None])
        assert result.shape == frame.shape


# ---------------------------------------------------------------------------
# 6. _parse_pipeline_summary
# ---------------------------------------------------------------------------

class TestParsePipelineSummary:
    _SAMPLE_OUTPUT = """
=================================================================
  PROCESSING COMPLETED SUMMARY
=================================================================
  Total Frames Processed : 460
  Overall Average FPS    : 4.9
  Total Unique Visitors  : 2
  Confirmed ENTRY Events : 5
  Confirmed EXIT Events  : 3
=================================================================
"""

    def test_full_output_parsed_correctly(self):
        s = _parse_pipeline_summary(self._SAMPLE_OUTPUT)
        assert s["frames_processed"] == 460
        assert s["unique_visitors"] == 2
        assert s["entry_events"] == 5
        assert s["exit_events"] == 3
        assert abs(s["fps"] - 4.9) < 0.01

    def test_empty_stdout_returns_zeros(self):
        s = _parse_pipeline_summary("")
        assert s["frames_processed"] == 0
        assert s["unique_visitors"] == 0
        assert s["entry_events"] == 0
        assert s["exit_events"] == 0
        assert s["fps"] == 0.0

    def test_partial_output_parsed(self):
        partial = "  Total Unique Visitors  : 7\n  Confirmed ENTRY Events : 3"
        s = _parse_pipeline_summary(partial)
        assert s["unique_visitors"] == 7
        assert s["entry_events"] == 3
        assert s["frames_processed"] == 0  # missing → default

    def test_fps_parsed_as_float(self):
        s = _parse_pipeline_summary("  Overall Average FPS    : 12.3")
        assert isinstance(s["fps"], float)
        assert abs(s["fps"] - 12.3) < 0.01

    def test_session_output_parsed_preferentially(self):
        stdout = """
  Total Frames Processed : 1502
  Overall Average FPS    : 8.2
  Session Unique Visitors: 2
  Session ENTRY Events   : 2
  Session EXIT Events    : 2
  Total Unique Visitors  : 5
  Confirmed ENTRY Events : 21
  Confirmed EXIT Events  : 16
"""
        s = _parse_pipeline_summary(stdout)
        assert s["unique_visitors"] == 2
        assert s["entry_events"] == 2
        assert s["exit_events"] == 2
        assert s["total_db_visitors"] == 5
        assert s["total_db_entries"] == 21
        assert s["total_db_exits"] == 16


# ---------------------------------------------------------------------------
# 7. process_image_upload (mocked models)
# ---------------------------------------------------------------------------

class TestProcessImageUpload:
    def test_invalid_bytes_returns_error(self, tmp_path):
        mock_detector = MagicMock()
        result = process_image_upload(
            file_bytes=b"not an image",
            detector=mock_detector,
            recognizer=None,
            visitor_manager=None,
            scan_dir=tmp_path / "scans",
        )
        assert result["success"] is False
        assert result["error"] is not None
        assert result["face_count"] == 0
        mock_detector.detect.assert_not_called()

    def test_no_faces_detected(self, tmp_path):
        mock_detector = MagicMock()
        mock_detector.detect.return_value = []  # no detections
        result = process_image_upload(
            file_bytes=_make_jpeg_bytes(),
            detector=mock_detector,
            recognizer=None,
            visitor_manager=None,
            scan_dir=tmp_path / "scans",
        )
        assert result["success"] is True
        assert result["face_count"] == 0
        assert result["faces"] == []
        assert result["annotated_image"] is not None

    def test_face_recognized(self, tmp_path):
        """A face with a valid embedding is recognized via visitor_manager."""
        mock_detector = MagicMock()
        mock_detector.detect.return_value = [([10, 10, 60, 60], 0.92)]

        mock_recognizer = MagicMock()
        dummy_emb = np.random.randn(512).astype(np.float32)
        mock_recognizer.extract_embedding.return_value = dummy_emb

        mock_vm = MagicMock()
        mock_vm.register_or_recognize.return_value = ("FACE_001", False, 0.97)

        result = process_image_upload(
            file_bytes=_make_jpeg_bytes(200, 200),
            detector=mock_detector,
            recognizer=mock_recognizer,
            visitor_manager=mock_vm,
            scan_dir=tmp_path / "scans",
        )

        assert result["success"] is True
        assert result["face_count"] == 1
        face = result["faces"][0]
        assert face["face_id"] == "FACE_001"
        assert face["is_new"] is False
        assert abs(face["similarity"] - 0.97) < 0.001
        mock_vm.register_or_recognize.assert_called_once_with(dummy_emb)

    def test_new_visitor_registered(self, tmp_path):
        """A new face triggers registration (is_new=True) without creating events."""
        mock_detector = MagicMock()
        mock_detector.detect.return_value = [([5, 5, 70, 70], 0.88)]

        mock_recognizer = MagicMock()
        mock_recognizer.extract_embedding.return_value = np.ones(512, dtype=np.float32)

        mock_vm = MagicMock()
        mock_vm.register_or_recognize.return_value = ("FACE_002", True, 1.0)

        result = process_image_upload(
            file_bytes=_make_jpeg_bytes(200, 200),
            detector=mock_detector,
            recognizer=mock_recognizer,
            visitor_manager=mock_vm,
            scan_dir=tmp_path / "scans",
        )

        assert result["success"] is True
        face = result["faces"][0]
        assert face["face_id"] == "FACE_002"
        assert face["is_new"] is True

    def test_embedding_failure_leaves_face_id_none(self, tmp_path):
        """If embedding extraction returns None, face_id remains None."""
        mock_detector = MagicMock()
        mock_detector.detect.return_value = [([10, 10, 50, 50], 0.80)]

        mock_recognizer = MagicMock()
        mock_recognizer.extract_embedding.return_value = None  # failed

        mock_vm = MagicMock()

        result = process_image_upload(
            file_bytes=_make_jpeg_bytes(200, 200),
            detector=mock_detector,
            recognizer=mock_recognizer,
            visitor_manager=mock_vm,
            scan_dir=tmp_path / "scans",
        )

        assert result["success"] is True
        assert result["faces"][0]["face_id"] is None
        mock_vm.register_or_recognize.assert_not_called()

    def test_annotated_image_is_rgb(self, tmp_path):
        """The returned annotated_image must be RGB (for st.image compatibility)."""
        mock_detector = MagicMock()
        mock_detector.detect.return_value = [([5, 5, 40, 40], 0.75)]

        mock_recognizer = MagicMock()
        mock_recognizer.extract_embedding.return_value = None

        result = process_image_upload(
            file_bytes=_make_jpeg_bytes(100, 100),
            detector=mock_detector,
            recognizer=mock_recognizer,
            visitor_manager=None,
            scan_dir=tmp_path / "scans",
        )
        # Streamlit's st.image expects RGB; verify the dtype and shape
        img = result["annotated_image"]
        assert img is not None
        assert img.ndim == 3
        assert img.shape[2] == 3


# ---------------------------------------------------------------------------
# 8. run_video_pipeline (subprocess mocked)
# ---------------------------------------------------------------------------

class TestRunVideoPipeline:
    _GOOD_STDOUT = (
        "\n=================================================================\n"
        "  PROCESSING COMPLETED SUMMARY\n"
        "=================================================================\n"
        "  Total Frames Processed : 120\n"
        "  Overall Average FPS    : 5.2\n"
        "  Total Unique Visitors  : 1\n"
        "  Confirmed ENTRY Events : 2\n"
        "  Confirmed EXIT Events  : 1\n"
        "=================================================================\n"
    )

    def test_successful_pipeline_parsed(self, tmp_path):
        video_path = tmp_path / "clip.mp4"
        video_path.write_bytes(b"")

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = self._GOOD_STDOUT
        mock_proc.stderr = ""

        with patch("src.upload_processor.subprocess.run", return_value=mock_proc):
            from src.upload_processor import run_video_pipeline
            result = run_video_pipeline(video_path, tmp_path)

        assert result["success"] is True
        assert result["frames_processed"] == 120
        assert result["unique_visitors"] == 1
        assert result["entry_events"] == 2
        assert result["exit_events"] == 1
        assert abs(result["fps"] - 5.2) < 0.01
        assert result["error"] is None

    def test_nonzero_returncode_with_summary_is_success(self, tmp_path):
        """returncode != 0 is treated as success when the summary was written."""
        video_path = tmp_path / "clip.mp4"
        video_path.write_bytes(b"")

        mock_proc = MagicMock()
        mock_proc.returncode = 1  # PowerShell NativeCommandError artifact
        mock_proc.stdout = self._GOOD_STDOUT
        mock_proc.stderr = "some stderr noise"

        with patch("src.upload_processor.subprocess.run", return_value=mock_proc):
            from src.upload_processor import run_video_pipeline
            result = run_video_pipeline(video_path, tmp_path)

        assert result["success"] is True
        assert result["frames_processed"] == 120

    def test_timeout_returns_failure(self, tmp_path):
        import subprocess
        video_path = tmp_path / "clip.mp4"
        video_path.write_bytes(b"")

        with patch(
            "src.upload_processor.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="main.py", timeout=900),
        ):
            from src.upload_processor import run_video_pipeline
            result = run_video_pipeline(video_path, tmp_path)

        assert result["success"] is False
        assert "timed out" in result["error"].lower()
        assert result["frames_processed"] == 0

    def test_exception_returns_failure(self, tmp_path):
        video_path = tmp_path / "clip.mp4"
        video_path.write_bytes(b"")

        with patch(
            "src.upload_processor.subprocess.run",
            side_effect=OSError("process failed"),
        ):
            from src.upload_processor import run_video_pipeline
            result = run_video_pipeline(video_path, tmp_path)

        assert result["success"] is False
        assert result["error"] is not None
