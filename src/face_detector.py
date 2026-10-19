"""Face detection module using face-specific YOLO architecture for Katomaran Face Tracker.

Uses Ultralytics YOLO with weights trained specifically for face detection (e.g. yolov8n-face.pt).
Clips bounding boxes to image dimensions, filters by confidence and minimum face area,
and provides download helpers and diagnostic messages for missing weights.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger("katomaran.face_detector")

# Known open-source face-trained YOLO releases
YOLO_FACE_DOWNLOAD_URL = (
    "https://huggingface.co/ElenaRyumina/MASAI_models/resolve/main/yolov8n-face.pt"
)


class FaceDetector:
    """YOLO-based face detection engine."""

    def __init__(
        self,
        model_path: str | Path,
        confidence_threshold: float = 0.40,
        min_face_size: int = 20,
        device: str = "cpu",
        auto_download: bool = True,
    ):
        """Initialize FaceDetector.

        Args:
            model_path: Path to the face detector model weights (.pt or .onnx).
            confidence_threshold: Minimum detection confidence score (0.0 to 1.0).
            min_face_size: Minimum pixel dimension (width and height) to keep detection.
            device: Inference device ('cpu', 'cuda', '0').
            auto_download: If True, attempts to download face weights if missing.
        """
        self.model_path = Path(model_path).resolve()
        self.confidence_threshold = confidence_threshold
        self.min_face_size = min_face_size
        self.device = device
        self.model = None

        self._load_model(auto_download=auto_download)

    def _load_model(self, auto_download: bool = True) -> None:
        """Load YOLO model weights, with optional automated download."""
        if not self.model_path.exists():
            if auto_download:
                logger.info(
                    f"Face detector model not found at '{self.model_path}'. Attempting download..."
                )
                success = self.download_weights(self.model_path)
                if not success:
                    raise FileNotFoundError(
                        f"Face detector model weights missing at: {self.model_path}.\n"
                        f"Please download face weights (e.g., yolov8n-face.pt) from:\n"
                        f"{YOLO_FACE_DOWNLOAD_URL}\n"
                        f"and place them at '{self.model_path}'."
                    )
            else:
                raise FileNotFoundError(
                    f"Face detector model weights missing at: {self.model_path}."
                )

        logger.info(f"Loading YOLO face detector from '{self.model_path}'...")
        try:
            from ultralytics import YOLO
            self.model = YOLO(str(self.model_path))
            logger.info("YOLO face detector loaded successfully.")
        except ImportError:
            raise ImportError(
                "Ultralytics library is required for YOLO face detection. Run: pip install ultralytics"
            )
        except Exception as e:
            logger.error(f"Failed to load YOLO model from '{self.model_path}': {e}", exc_info=True)
            raise

    @staticmethod
    def download_weights(dest_path: Path) -> bool:
        """Download open-source face-trained YOLO model weights."""
        import urllib.request
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            logger.info(f"Downloading yolov8n-face.pt from {YOLO_FACE_DOWNLOAD_URL}...")
            urllib.request.urlretrieve(YOLO_FACE_DOWNLOAD_URL, str(dest_path))
            if dest_path.exists() and dest_path.stat().st_size > 10000:
                logger.info(f"Downloaded weights successfully ({dest_path.stat().st_size / (1024*1024):.2f} MB).")
                return True
            return False
        except Exception as e:
            logger.warning(f"Automated weights download failed: {e}")
            return False

    def detect(
        self, frame: np.ndarray
    ) -> List[Tuple[List[float], float]]:
        """Run face detection inference on a BGR video frame.

        Args:
            frame: OpenCV BGR image (H x W x C).

        Returns:
            List of tuples: (bbox [x1, y1, x2, y2], confidence).
            Coordinates are clipped to frame boundaries and filtered by confidence & minimum size.
        """
        if self.model is None or frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]

        # Run inference without verbose stdout spam
        results = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False,
        )

        detections: List[Tuple[List[float], float]] = []

        if not results:
            return detections

        for r in results:
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for box in boxes:
                conf = float(box.conf[0].item())
                if conf < self.confidence_threshold:
                    continue

                # Bounding box in [x1, y1, x2, y2]
                xyxy = box.xyxy[0].tolist()
                x1 = max(0.0, float(xyxy[0]))
                y1 = max(0.0, float(xyxy[1]))
                x2 = min(float(w), float(xyxy[2]))
                y2 = min(float(h), float(xyxy[3]))

                bw = x2 - x1
                bh = y2 - y1

                # Filter out excessively small or invalid bounding boxes
                if bw < self.min_face_size or bh < self.min_face_size:
                    continue

                detections.append(([x1, y1, x2, y2], conf))

        return detections
