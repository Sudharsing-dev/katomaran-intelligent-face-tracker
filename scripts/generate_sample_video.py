"""Sample video generator using standard open-source face crops from Ultralytics benchmark images.

Extracts two distinct real human face crops (Visitor A and Visitor B) and generates
a comprehensive 460-frame benchmark video 'videos/sample_video.mp4':
- Scenario 1 (Frames 0-90): Visitor A enters from left, traverses scene, exits right.
- Scenario 2 (Frames 91-140): Empty scene (verifies grace period expiration & EXIT event confirmation).
- Scenario 3 (Frames 141-230): Visitor B enters from right, traverses scene, exits left.
- Scenario 4 (Frames 231-280): Empty scene (verifies Visitor B EXIT confirmation).
- Scenario 5 (Frames 281-370): Returning Visitor A re-enters (verifies ArcFace re-identification & count integrity).
- Scenario 6 (Frames 371-460): Simultaneous entry of Visitor A and Visitor B (verifies multi-face tracking).

Usage:
    python scripts/generate_sample_video.py
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import cv2
import numpy as np
import ultralytics


def get_face_crops() -> tuple[np.ndarray, np.ndarray]:
    """Extract real human face crops from open-source zidane.jpg benchmark."""
    assets_dir = os.path.join(os.path.dirname(ultralytics.__file__), "assets")
    zidane_path = os.path.join(assets_dir, "zidane.jpg")
    img = cv2.imread(zidane_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load benchmark image from {zidane_path}")

    # Face 1 (Zidane face center right)
    crop_a = img[92:276, 913:1055]
    # Face 2 (Secondary person left)
    crop_b = img[242:448, 542:663]

    # Resize crops to standard size (e.g. 100x120) with smooth borders
    crop_a = cv2.resize(crop_a, (100, 120), interpolation=cv2.INTER_AREA)
    crop_b = cv2.resize(crop_b, (100, 120), interpolation=cv2.INTER_AREA)

    return crop_a, crop_b


def overlay_crop(
    canvas: np.ndarray,
    crop: np.ndarray,
    center_x: int,
    center_y: int,
) -> None:
    """Safely overlay a face crop onto the canvas frame."""
    ch, cw = crop.shape[:2]
    h, w = canvas.shape[:2]

    x1 = int(center_x - cw // 2)
    y1 = int(center_y - ch // 2)
    x2 = x1 + cw
    y2 = y1 + ch

    if x2 <= 0 or y2 <= 0 or x1 >= w or y1 >= h:
        return

    # Clipping
    src_x1 = max(0, -x1)
    src_y1 = max(0, -y1)
    src_x2 = cw - max(0, x2 - w)
    src_y2 = ch - max(0, y2 - h)

    dst_x1 = max(0, x1)
    dst_y1 = max(0, y1)
    dst_x2 = min(w, x2)
    dst_y2 = min(h, y2)

    if dst_x2 > dst_x1 and dst_y2 > dst_y1:
        canvas[dst_y1:dst_y2, dst_x1:dst_x2] = crop[src_y1:src_y2, src_x1:src_x2]


def generate_video(
    output_path: str = "videos/sample_video.mp4",
    width: int = 960,
    height: int = 540,
    fps: int = 30,
) -> Path:
    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    face_a, face_b = get_face_crops()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_file), fourcc, fps, (width, height))

    total_frames = 460
    print(f"Generating realistic sample video at {out_file} ({total_frames} frames)...")

    for f in range(total_frames):
        # Create realistic office hallway background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = (230, 230, 225)

        # Architectural lines
        cv2.line(frame, (0, int(height * 0.7)), (width, int(height * 0.7)), (180, 180, 175), 2)
        cv2.rectangle(frame, (100, 50), (220, int(height * 0.7)), (210, 210, 205), -1)
        cv2.rectangle(frame, (100, 50), (220, int(height * 0.7)), (160, 160, 155), 2)

        scenario_text = ""

        # Scenario 1 (0-90): Visitor A enters and traverses
        if 0 <= f < 90:
            scenario_text = "Scenario 1: Visitor A Enters (New Registration -> FACE_001)"
            progress = f / 89.0
            x = int(80 + progress * (width - 160))
            y = int(height * 0.45 + math.sin(f * 0.1) * 10)
            overlay_crop(frame, face_a, x, y)

        # Scenario 2 (90-140): Empty scene (Grace period & confirm EXIT)
        elif 90 <= f < 140:
            scenario_text = "Scenario 2: Scene Empty (Grace Period -> Confirms EXIT for FACE_001)"

        # Scenario 3 (140-230): Visitor B enters from right
        elif 140 <= f < 230:
            scenario_text = "Scenario 3: Visitor B Enters (New Registration -> FACE_002)"
            progress = (f - 140) / 89.0
            x = int((width - 80) - progress * (width - 160))
            y = int(height * 0.48 + math.sin(f * 0.1) * 8)
            overlay_crop(frame, face_b, x, y)

        # Scenario 4 (230-280): Empty scene (Confirm Visitor B EXIT)
        elif 230 <= f < 280:
            scenario_text = "Scenario 4: Scene Empty (Confirms EXIT for FACE_002)"

        # Scenario 5 (280-370): Returning Visitor A (Re-identification)
        elif 280 <= f < 370:
            scenario_text = "Scenario 5: Returning Visitor A (ArcFace Match -> Reuses FACE_001)"
            progress = (f - 280) / 89.0
            x = int(80 + progress * (width - 160))
            y = int(height * 0.45 + math.sin(f * 0.1) * 10)
            overlay_crop(frame, face_a, x, y)

        # Scenario 6 (370-460): Simultaneous entry of Visitor A and Visitor B
        elif 370 <= f < 460:
            scenario_text = "Scenario 6: Multi-Face Tracking (A & B Together)"
            xA = int(120 + ((f - 370) / 90.0) * (width - 240))
            yA = int(height * 0.40)
            overlay_crop(frame, face_a, xA, yA)

            xB = int((width - 120) - ((f - 370) / 90.0) * (width - 240))
            yB = int(height * 0.52)
            overlay_crop(frame, face_b, xB, yB)

        # Header bar
        cv2.rectangle(frame, (0, 0), (width, 36), (35, 35, 35), -1)
        cv2.putText(
            frame,
            f"FRAME {f:03d}/{total_frames} | {scenario_text}",
            (15, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        writer.write(frame)

    writer.release()
    print(f"Generated realistic sample video: {out_file} ({out_file.stat().st_size / (1024*1024):.2f} MB)")
    return out_file


if __name__ == "__main__":
    generate_video()
