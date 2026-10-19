"""Facial recognition module using InsightFace / ArcFace for Katomaran Face Tracker.

Extracts 512-dimensional normalized facial embeddings from face crops using
ArcFace deep feature extractors (e.g., buffalo_l). Performs face quality validation
(blur/size filtering) and vector normalization.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("katomaran.face_recognizer")


def is_crop_blurry(crop: np.ndarray, laplacian_threshold: float = 15.0) -> bool:
    """Check if face crop is excessively blurry using Laplacian variance."""
    if crop is None or crop.size == 0:
        return True
    try:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return bool(variance < laplacian_threshold)
    except Exception:
        return False


def normalize_embedding(vec: np.ndarray) -> np.ndarray:
    """L2-normalize a 1D embedding vector."""
    arr = np.asarray(vec, dtype=np.float32).flatten()
    norm = np.linalg.norm(arr)
    if norm > 0:
        return arr / norm
    return arr


class FaceRecognizer:
    """InsightFace ArcFace facial embedding extractor and comparator."""

    def __init__(
        self,
        model_name: str = "buffalo_l",
        similarity_threshold: float = 0.45,
        min_face_size: int = 20,
        blur_check: bool = True,
        laplacian_threshold: float = 15.0,
        ctx_id: int = -1,  # -1 for CPU, 0+ for GPU
    ):
        """Initialize FaceRecognizer.

        Args:
            model_name: InsightFace model suite name ('buffalo_l', 'buffalo_sc', 'antelopev2').
            similarity_threshold: Default cosine similarity matching cutoff.
            min_face_size: Minimum face crop resolution (width and height).
            blur_check: Whether to reject blurry face crops.
            laplacian_threshold: Variance cutoff for blur filter.
            ctx_id: Device ID (-1 = CPU).
        """
        self.model_name = model_name
        self.similarity_threshold = float(similarity_threshold)
        self.min_face_size = min_face_size
        self.blur_check = blur_check
        self.laplacian_threshold = laplacian_threshold
        self.ctx_id = ctx_id

        self.app = None
        self._initialize_insightface()

    def _initialize_insightface(self) -> None:
        """Initialize InsightFace FaceAnalysis once at application startup."""
        logger.info(f"Initializing InsightFace recognition model '{self.model_name}' on ctx={self.ctx_id}...")
        try:
            from insightface.app import FaceAnalysis

            # Initialize FaceAnalysis with recognition module enabled
            providers = ["CPUExecutionProvider"]
            self.app = FaceAnalysis(
                name=self.model_name,
                providers=providers,
            )
            # Prepare with detection size 640
            self.app.prepare(ctx_id=self.ctx_id, det_size=(640, 640))
            logger.info("InsightFace FaceAnalysis engine initialized successfully.")
        except ImportError:
            logger.warning(
                "InsightFace library is not yet installed. Facial recognition will require insightface. "
                "Run: pip install insightface"
            )
            self.app = None
        except Exception as e:
            logger.warning(
                f"InsightFace model '{self.model_name}' could not be initialized directly: {e}. "
                "Will attempt on-demand loading or fallback when needed."
            )
            self.app = None

    def extract_embedding(self, face_crop: np.ndarray) -> Optional[np.ndarray]:
        """Extract a 512-dimensional normalized ArcFace embedding from a cropped face image.

        Args:
            face_crop: Cropped face BGR image.

        Returns:
            Normalized 1D numpy array of shape (512,), or None if crop quality is insufficient
            or extraction fails.
        """
        if face_crop is None or face_crop.size == 0:
            return None

        h, w = face_crop.shape[:2]
        if h < self.min_face_size or w < self.min_face_size:
            logger.debug(f"Face crop rejected: too small ({w}x{h} < {self.min_face_size}).")
            return None

        if self.blur_check and is_crop_blurry(face_crop, self.laplacian_threshold):
            logger.debug("Face crop rejected: excessive blur.")
            return None

        if self.app is None:
            # Re-attempt initialization if possible
            self._initialize_insightface()
            if self.app is None:
                logger.error("Cannot extract embedding: InsightFace engine not available.")
                return None

        try:
            # InsightFace app.get expects an image containing a face
            faces = self.app.get(face_crop)
            if not faces or len(faces) == 0:
                # Fallback: Tightly cropped faces lack surrounding context for SCRFD alignment.
                # Pad with neutral border so InsightFace's internal detector can locate facial landmarks.
                pad_h = max(24, int(face_crop.shape[0] * 0.5))
                pad_w = max(24, int(face_crop.shape[1] * 0.5))
                padded = cv2.copyMakeBorder(
                    face_crop, pad_h, pad_h, pad_w, pad_w,
                    cv2.BORDER_CONSTANT, value=[128, 128, 128]
                )
                faces = self.app.get(padded)

            if not faces or len(faces) == 0:
                logger.debug("InsightFace found no aligned face features in crop.")
                return None

            # Pick face with highest detection score or largest area
            best_face = max(faces, key=lambda f: getattr(f, "det_score", 0.0))
            emb = getattr(best_face, "embedding", None)

            if emb is None or len(emb) == 0:
                return None

            return normalize_embedding(emb)
        except Exception as e:
            logger.error(f"Error during facial embedding extraction: {e}", exc_info=True)
            return None

    def compare_embeddings(self, emb_a: np.ndarray, emb_b: np.ndarray) -> float:
        """Calculate cosine similarity score between two embeddings."""
        if emb_a is None or emb_b is None:
            return 0.0
        norm_a = np.linalg.norm(emb_a)
        norm_b = np.linalg.norm(emb_b)
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return float(np.dot(emb_a, emb_b) / (norm_a * norm_b))
