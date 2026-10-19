"""Upload processing utilities for the Katomaran Intelligent Face Tracker dashboard.

Image uploads
-------------
A single still image is processed through YOLO face detection and optional
ArcFace recognition/registration.  The existing ENTRY/EXIT event model is
designed around a temporal sequence of tracked frames and is therefore NOT
invoked for still images.  This module:

  * Runs YOLO face detection and draws annotated bounding boxes for display.
  * Extracts ArcFace embeddings and calls ``VisitorManager.register_or_recognize``
    to match or register each face — the visitor gallery and ``visitors`` table
    are updated but no ``events`` rows are created.
  * Saves detected face crops to ``logs/image_scans/<YYYY-MM-DD>/`` — a separate
    directory from ``logs/entries/`` and ``logs/exits/`` so gallery integrity is
    maintained and entries are never fabricated from static frames.

Video uploads
-------------
Uploaded videos are saved to ``videos/uploads/`` with a UUID-prefixed unique
filename and are then processed by calling the existing ``main.py`` pipeline as
a headless subprocess.  The complete stack — YOLO, InsightFace, FaceTracker,
VisitorManager, EventManager, ImageStorage, DatabaseManager — is reused
unchanged, exactly as it runs from the terminal.
"""

from __future__ import annotations

import logging
import re
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.image_storage import crop_face

logger = logging.getLogger("katomaran.upload_processor")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_IMAGE_PIXELS = 4096          # down-scale images larger than this dimension
ALLOWED_IMAGE_EXTS = {"jpg", "jpeg", "png"}
ALLOWED_VIDEO_EXTS = {"mp4", "avi", "mov"}
VIDEO_PIPELINE_TIMEOUT_SEC = 900  # 15-minute hard limit for subprocess


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_and_decode_image(file_bytes: bytes) -> Optional[np.ndarray]:
    """Decode raw image bytes to an OpenCV BGR frame.

    Returns ``None`` when the bytes are empty, corrupt, or cannot be decoded
    by OpenCV.  Very large images are down-scaled to ``MAX_IMAGE_PIXELS`` on
    their longest dimension before being returned.
    """
    if not file_bytes:
        return None
    try:
        arr = np.frombuffer(file_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None or frame.size == 0:
            return None
        h, w = frame.shape[:2]
        if max(h, w) > MAX_IMAGE_PIXELS:
            scale = MAX_IMAGE_PIXELS / max(h, w)
            frame = cv2.resize(
                frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
            )
        return frame
    except Exception as exc:
        logger.warning(f"Image decode failed: {exc}")
        return None


def is_safe_upload_filename(name: str) -> bool:
    """Return ``True`` if *name* is a safe, flat filename with an allowed extension.

    Rejects names that contain directory separators (path-traversal guard) and
    extensions that are not in the allowed image / video sets.
    """
    p = Path(name)
    # Reject any path that has a parent component other than "."
    if p.parent != Path("."):
        return False
    ext = p.suffix.lower().lstrip(".")
    return ext in ALLOWED_IMAGE_EXTS | ALLOWED_VIDEO_EXTS


def safe_unique_video_filename(original_name: str) -> str:
    """Return a UUID-prefixed, filesystem-safe video filename.

    Example: ``a3f9c12b01_my_video.mp4``
    """
    p = Path(original_name)
    suffix = p.suffix.lower() if p.suffix else ".mp4"
    stem = re.sub(r"[^\w\-]", "_", p.stem)[:40]
    uid = uuid.uuid4().hex[:10]
    return f"{uid}_{stem}{suffix}"


# ---------------------------------------------------------------------------
# Image annotation
# ---------------------------------------------------------------------------

def draw_detections(
    frame: np.ndarray,
    detections: List[Tuple[List[float], float]],
    face_ids: Optional[List[Optional[str]]] = None,
) -> np.ndarray:
    """Draw bounding boxes and optional face IDs on a copy of *frame* (BGR).

    Green boxes indicate a recognized / registered identity; orange boxes
    indicate that recognition was unavailable or the embedding failed.
    """
    out = frame.copy()
    for idx, (bbox, conf) in enumerate(detections):
        x1, y1, x2, y2 = [int(round(c)) for c in bbox]
        fid = face_ids[idx] if face_ids and idx < len(face_ids) else None
        color = (0, 200, 0) if fid else (0, 140, 255)  # green / orange
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{fid or 'Unknown'} ({conf:.2f})"
        label_y = max(14, y1 - 8)
        cv2.putText(out, label, (x1 + 2, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------------------
# Image processing pipeline
# ---------------------------------------------------------------------------

def process_image_upload(
    file_bytes: bytes,
    detector: Any,            # src.face_detector.FaceDetector
    recognizer: Any,          # src.face_recognizer.FaceRecognizer | None
    visitor_manager: Any,     # src.visitor_manager.VisitorManager | None
    scan_dir: Path,
) -> Dict[str, Any]:
    """Run face detection and optional recognition on an uploaded still image.

    Args:
        file_bytes:      Raw bytes of the uploaded image.
        detector:        Initialised ``FaceDetector`` instance.
        recognizer:      Initialised ``FaceRecognizer`` or ``None``.
        visitor_manager: Initialised ``VisitorManager`` or ``None``.
        scan_dir:        Directory under which face crops are saved
                         (``logs/image_scans``).

    Returns:
        Dict with keys:

        * ``success``         – ``bool``
        * ``error``           – ``str | None``
        * ``annotated_image`` – RGB ``np.ndarray`` suitable for ``st.image``
        * ``face_count``      – number of detected faces
        * ``faces``           – ``list[dict]`` — one entry per detected face:

          - ``bbox``         ``[x1, y1, x2, y2]``
          - ``confidence``   ``float``
          - ``face_id``      ``str | None``
          - ``is_new``       ``bool``
          - ``similarity``   ``float``
          - ``crop_path``    ``str | None``

    Architecture note
    -----------------
    ENTRY / EXIT events are **not** created.  ``VisitorManager.register_or_recognize``
    updates only the ``visitors`` table; no ``events`` rows are written.
    """
    frame = validate_and_decode_image(file_bytes)
    if frame is None:
        return {
            "success": False,
            "error": "Could not decode the uploaded image. Ensure it is a valid JPG or PNG file.",
            "annotated_image": None,
            "face_count": 0,
            "faces": [],
        }

    # YOLO detection
    detections: List[Tuple[List[float], float]] = detector.detect(frame)

    faces: List[Dict[str, Any]] = []
    face_ids: List[Optional[str]] = []

    for bbox, conf in detections:
        entry: Dict[str, Any] = {
            "bbox": bbox,
            "confidence": conf,
            "face_id": None,
            "is_new": False,
            "similarity": 0.0,
            "crop_path": None,
        }

        if recognizer is not None and visitor_manager is not None:
            face_crop = crop_face(frame, bbox)
            if face_crop is not None:
                emb = recognizer.extract_embedding(face_crop)
                if emb is not None:
                    face_id, is_new, sim = visitor_manager.register_or_recognize(emb)
                    entry["face_id"] = face_id
                    entry["is_new"] = is_new
                    entry["similarity"] = sim

                    # Save crop to dedicated scan directory
                    dt = datetime.now(timezone.utc)
                    date_str = dt.strftime("%Y-%m-%d")
                    time_str = dt.strftime("%Y%m%d_%H%M%S_%f")
                    dest_dir = scan_dir / date_str
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    safe_fid = re.sub(r"[^\w\-]", "_", face_id)
                    crop_file = dest_dir / f"{safe_fid}_{time_str}.jpg"
                    counter = 1
                    while crop_file.exists():
                        crop_file = dest_dir / f"{safe_fid}_{time_str}_{counter}.jpg"
                        counter += 1
                    try:
                        cv2.imwrite(str(crop_file), face_crop)
                        entry["crop_path"] = str(crop_file)
                        logger.info(f"Saved image scan crop: {crop_file.name}")
                    except Exception as exc:
                        logger.warning(f"Could not save scan crop: {exc}")

        face_ids.append(entry["face_id"])
        faces.append(entry)

    # Annotate and convert BGR→RGB for st.image
    annotated_bgr = draw_detections(frame, detections, face_ids)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    return {
        "success": True,
        "error": None,
        "annotated_image": annotated_rgb,
        "face_count": len(detections),
        "faces": faces,
    }


# ---------------------------------------------------------------------------
# Video processing pipeline
# ---------------------------------------------------------------------------

def save_uploaded_video(
    file_bytes: bytes,
    original_name: str,
    upload_dir: Path,
) -> Path:
    """Write *file_bytes* to *upload_dir* under a safe unique filename.

    Args:
        file_bytes:    Raw video bytes from the Streamlit uploader.
        original_name: Original filename (used to preserve extension).
        upload_dir:    Destination directory (created if absent).

    Returns:
        ``Path`` to the saved file.

    Raises:
        ``OSError`` on write failure.
    """
    upload_dir.mkdir(parents=True, exist_ok=True)
    filename = safe_unique_video_filename(original_name)
    dest = upload_dir / filename
    dest.write_bytes(file_bytes)
    size_mb = len(file_bytes) / (1024 * 1024)
    logger.info(f"Saved uploaded video '{original_name}' → '{dest}' ({size_mb:.2f} MB).")
    return dest


def _parse_pipeline_summary(stdout: str) -> Dict[str, Any]:
    """Extract numeric statistics from the ``main.py`` stdout summary block."""
    result: Dict[str, Any] = {
        "frames_processed": 0,
        "unique_visitors": 0,
        "entry_events": 0,
        "exit_events": 0,
        "fps": 0.0,
        "total_db_visitors": 0,
        "total_db_entries": 0,
        "total_db_exits": 0,
    }

    m_frames = re.search(r"Total Frames Processed\s*:\s*(\d+)", stdout)
    if m_frames:
        result["frames_processed"] = int(m_frames.group(1))

    m_fps = re.search(r"Overall Average FPS\s*:\s*([\d.]+)", stdout)
    if m_fps:
        result["fps"] = float(m_fps.group(1))

    # Video session stats (specific to this video run)
    m_sess_vis = re.search(r"Session Unique Visitors\s*:\s*(\d+)", stdout)
    m_sess_ent = re.search(r"Session ENTRY Events\s*:\s*(\d+)", stdout)
    m_sess_ext = re.search(r"Session EXIT Events\s*:\s*(\d+)", stdout)

    # Database cumulative totals
    m_db_vis = re.search(r"Total Unique Visitors\s*:\s*(\d+)", stdout)
    m_db_ent = re.search(r"Confirmed ENTRY Events\s*:\s*(\d+)", stdout)
    m_db_ext = re.search(r"Confirmed EXIT Events\s*:\s*(\d+)", stdout)

    if m_sess_vis:
        result["unique_visitors"] = int(m_sess_vis.group(1))
    elif m_db_vis:
        result["unique_visitors"] = int(m_db_vis.group(1))

    if m_sess_ent:
        result["entry_events"] = int(m_sess_ent.group(1))
    elif m_db_ent:
        result["entry_events"] = int(m_db_ent.group(1))

    if m_sess_ext:
        result["exit_events"] = int(m_sess_ext.group(1))
    elif m_db_ext:
        result["exit_events"] = int(m_db_ext.group(1))

    if m_db_vis:
        result["total_db_visitors"] = int(m_db_vis.group(1))
    if m_db_ent:
        result["total_db_entries"] = int(m_db_ent.group(1))
    if m_db_ext:
        result["total_db_exits"] = int(m_db_ext.group(1))

    return result


def run_video_pipeline(video_path: Path, project_root: Path) -> Dict[str, Any]:
    """Process an uploaded video through the existing ``main.py`` pipeline.

    Invokes ``main.py --source <path> --no-display`` as a subprocess using the
    same Python interpreter (``sys.executable``), so the virtual-environment
    packages and installed models are inherited correctly.

    The complete existing stack is reused:
    YOLO detection → InsightFace/ArcFace → FaceTracker → VisitorManager →
    EventManager → ImageStorage → DatabaseManager → ``logs/events.log``.

    Args:
        video_path:   Absolute ``Path`` to the saved upload.
        project_root: Project root directory (contains ``main.py``).

    Returns:
        Dict with keys: success, error, returncode, stdout, stderr,
        frames_processed, unique_visitors, entry_events, exit_events, fps,
        video_path.
    """
    _empty: Dict[str, Any] = {
        "frames_processed": 0,
        "unique_visitors": 0,
        "entry_events": 0,
        "exit_events": 0,
        "fps": 0.0,
    }

    cmd = [
        sys.executable,
        str(project_root / "main.py"),
        "--source", str(video_path),
        "--no-display",
    ]
    logger.info(f"Launching pipeline subprocess: {' '.join(cmd)}")

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(project_root),
            timeout=VIDEO_PIPELINE_TIMEOUT_SEC,
        )
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        summary = _parse_pipeline_summary(stdout)

        # Treat as success when the summary block was written to stdout,
        # even if returncode is non-zero (PowerShell stderr artifact).
        pipeline_ok = (proc.returncode == 0) or (summary["frames_processed"] > 0)

        return {
            **summary,
            "success": pipeline_ok,
            "error": None if pipeline_ok else f"Pipeline exited with code {proc.returncode}.",
            "returncode": proc.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "video_path": str(video_path),
        }

    except subprocess.TimeoutExpired:
        msg = f"Processing timed out after {VIDEO_PIPELINE_TIMEOUT_SEC // 60} minutes."
        logger.error(msg)
        return {**_empty, "success": False, "error": msg,
                "returncode": -1, "stdout": "", "stderr": "",
                "video_path": str(video_path)}

    except Exception as exc:
        logger.error(f"Pipeline subprocess raised an exception: {exc}", exc_info=True)
        return {**_empty, "success": False, "error": str(exc),
                "returncode": -1, "stdout": "", "stderr": "",
                "video_path": str(video_path)}
