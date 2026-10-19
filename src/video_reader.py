"""Video stream and file reader module for Katomaran Intelligent Face Tracker.

Provides robust reading of local video files and live RTSP network camera streams
with automatic reconnection, resource cleanup, and video metadata extraction.

Connection behaviour
--------------------
- Local video files: opened once; EOF terminates the frame generator.
- RTSP / HTTP streams: opened with FFmpeg TCP transport.  If the initial open
  fails, ``_initial_rtsp_connect()`` retries up to ``max_reconnect_attempts``
  times, printing progress to the terminal and logging each attempt.  When all
  attempts are exhausted the ``connected`` property returns ``False`` and the
  frame generator yields nothing — **no visitor records or events are created**.
- Mid-stream disconnections: ``_reconnect_rtsp()`` is called from inside the
  frame generator and follows the same retry / give-up logic.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Generator, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("katomaran.video_reader")


class VideoReader:
    """Manages OpenCV VideoCapture for local files, RTSP streams, and webcams.

    For RTSP sources the constructor runs ``_initial_rtsp_connect()`` which
    retries up to ``max_reconnect_attempts`` times and prints clear progress
    information.  Check ``reader.connected`` before entering the frame loop to
    avoid processing when no camera is available.
    """

    def __init__(
        self,
        source: str | int,
        reconnect_delay_sec: float = 2.0,
        max_reconnect_attempts: int = 5,
        target_width: Optional[int] = 960,
    ):
        """Initialize VideoReader.

        Args:
            source: Path to video file, RTSP URL, or integer webcam index.
            reconnect_delay_sec: Seconds to wait before each RTSP reconnection.
            max_reconnect_attempts: Maximum consecutive reconnection attempts.
            target_width: Resize frames to this width (maintains aspect ratio).
                          Pass ``None`` to disable resizing.
        """
        self.source = source
        self.reconnect_delay_sec = reconnect_delay_sec
        self.max_reconnect_attempts = max_reconnect_attempts
        self.target_width = target_width

        self.cap: Optional[cv2.VideoCapture] = None
        self.is_rtsp = isinstance(source, str) and source.startswith(
            ("rtsp://", "rtsps://", "http://", "https://")
        )
        self.is_live = self.is_rtsp or isinstance(source, int)

        # Video metadata (populated by _open_source on success)
        self.fps: float = 30.0
        self.total_frames: int = 0
        self.width: int = 0
        self.height: int = 0
        self.current_frame_idx: int = 0

        # Internal connection flag — True only after cap.isOpened() succeeded
        self._connected: bool = False

        if self.is_rtsp:
            self._initial_rtsp_connect()
        else:
            self._open_source()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def connected(self) -> bool:
        """True if the source is open and ready to deliver frames."""
        return self._connected

    def get_sanitized_source_name(self) -> str:
        """Return the source string with embedded passwords replaced by ***:***."""
        src_str = str(self.source)
        if "@" in src_str and "://" in src_str:
            proto, rest = src_str.split("://", 1)
            _, host_part = rest.split("@", 1)
            return f"{proto}://***:***@{host_part}"
        return src_str

    def frames(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        """Yield ``(frame_index, bgr_frame)`` pairs until the stream ends.

        If the source was never connected (``self.connected`` is ``False``),
        the generator exits immediately without yielding — no visitor records
        or pipeline state is modified.
        """
        if not self._connected:
            logger.warning(
                "VideoReader.frames() called but source is not connected. "
                "No frames will be produced."
            )
            return

        # Safety net: cap may have been released externally
        if not self.cap or not self.cap.isOpened():
            if self.is_rtsp:
                if not self._reconnect_rtsp():
                    return
            else:
                return

        while True:
            success, frame = self.cap.read()
            if not success or frame is None or frame.size == 0:
                if self.is_rtsp:
                    if self._reconnect_rtsp():
                        continue
                    else:
                        break
                else:
                    logger.info(
                        f"Reached end of video file after frame #{self.current_frame_idx}."
                    )
                    break

            # Resize for faster downstream inference
            if self.target_width and frame.shape[1] > self.target_width:
                scale = self.target_width / frame.shape[1]
                new_h = int(frame.shape[0] * scale)
                frame = cv2.resize(
                    frame, (self.target_width, new_h), interpolation=cv2.INTER_AREA
                )

            yield self.current_frame_idx, frame
            self.current_frame_idx += 1

    def release(self) -> None:
        """Release the underlying OpenCV VideoCapture and reset the cap handle."""
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _open_source(self) -> bool:
        """Attempt a single open of the configured source.

        Updates ``self._connected``, ``self.width``, ``self.height``,
        ``self.fps``, and ``self.total_frames`` on success.
        """
        self.release()

        display_source = self.get_sanitized_source_name()
        logger.info(f"Opening video source: {display_source} (is_rtsp={self.is_rtsp})")

        # For local files, verify existence before calling VideoCapture
        if isinstance(self.source, str) and not self.is_rtsp:
            p = Path(self.source)
            if not p.exists():
                logger.error(f"Video file does not exist: {p}")
                self._connected = False
                return False

        # Open with FFmpeg backend for RTSP; default backend for files/webcams
        if self.is_rtsp:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
            self.cap = cv2.VideoCapture(str(self.source), cv2.CAP_FFMPEG)
        else:
            self.cap = cv2.VideoCapture(self.source)

        if not self.cap or not self.cap.isOpened():
            logger.warning(f"Failed to open '{display_source}'.")
            self._connected = False
            return False

        # Read stream / file properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        raw_fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.fps = raw_fps if (raw_fps > 0.0 and not np.isnan(raw_fps)) else 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        logger.info(
            f"Source opened: {self.width}x{self.height} @ {self.fps:.2f} FPS "
            f"(total_frames={self.total_frames})"
        )
        self._connected = True
        return True

    def _initial_rtsp_connect(self) -> bool:
        """Open an RTSP source at startup with terminal progress and an error summary.

        Retries up to ``max_reconnect_attempts`` times.  Credentials are never
        included in terminal output or log messages.

        Returns:
            ``True`` if connected within the allowed attempts.
        """
        display_source = self.get_sanitized_source_name()
        logger.info(
            f"Attempting initial RTSP connection to '{display_source}' "
            f"(up to {self.max_reconnect_attempts} attempt(s), "
            f"{self.reconnect_delay_sec}s delay between attempts)."
        )
        print(
            f"\n[RTSP] Connecting to : {display_source}\n"
            f"       Max attempts  : {self.max_reconnect_attempts}\n"
            f"       Retry delay   : {self.reconnect_delay_sec}s\n"
        )

        for attempt in range(1, self.max_reconnect_attempts + 1):
            print(
                f"  [RTSP] Attempt {attempt}/{self.max_reconnect_attempts}...",
                end=" ",
                flush=True,
            )
            logger.info(
                f"RTSP initial connection attempt {attempt}/{self.max_reconnect_attempts}."
            )

            if self._open_source():
                print("SUCCESS")
                logger.info(
                    f"RTSP stream connected on attempt {attempt}. "
                    f"Resolution: {self.width}x{self.height}, FPS: {self.fps:.2f}"
                )
                return True

            print("FAILED")

            if attempt < self.max_reconnect_attempts:
                print(
                    f"  [RTSP] Waiting {self.reconnect_delay_sec}s before "
                    f"attempt {attempt + 1}..."
                )
                logger.info(
                    f"Waiting {self.reconnect_delay_sec}s before attempt {attempt + 1}."
                )
                time.sleep(self.reconnect_delay_sec)

        # All attempts exhausted — log and print actionable checklist
        logger.error(
            f"RTSP startup failed: could not open '{display_source}' "
            f"after {self.max_reconnect_attempts} attempt(s). "
            "Verify camera URL, credentials, network connectivity, and firewall rules."
        )
        print(
            f"\n  ERROR: Unable to connect to the RTSP camera.\n"
            f"         Source   : {display_source}\n"
            f"         Attempts : {self.max_reconnect_attempts} (all failed)\n"
            f"\n"
            f"  Troubleshooting checklist:\n"
            f"    1. Is the camera powered on? (ping <CAMERA_IP>)\n"
            f"    2. Is the RTSP URL correct?\n"
            f"       Format: rtsp://USERNAME:PASSWORD@CAMERA_IP:PORT/STREAM_PATH\n"
            f"    3. Are the credentials correct?\n"
            f"    4. Is port 554 (or the configured port) open in the firewall?\n"
            f"    5. Open the URL in VLC → Media → Open Network Stream to verify.\n"
            f"    6. Ask your camera administrator for the correct URL.\n"
            f"\n"
            f"  The application will now exit without modifying any visitor data.\n"
        )
        self._connected = False
        return False

    def _reconnect_rtsp(self) -> bool:
        """Reconnect after a mid-stream RTSP failure.

        Uses the same retry count and delay as the initial connection handler.
        Called from inside the ``frames()`` generator when ``cap.read()`` fails.

        Returns:
            ``True`` if reconnected successfully; ``False`` when retries are
            exhausted (sets ``self._connected = False``).
        """
        display_source = self.get_sanitized_source_name()
        logger.warning(
            f"RTSP stream lost ('{display_source}'). "
            f"Attempting reconnection (up to {self.max_reconnect_attempts} time(s))..."
        )
        print(f"\n  [RTSP] Stream lost. Reconnecting to: {display_source}")

        for attempt in range(1, self.max_reconnect_attempts + 1):
            print(
                f"  [RTSP] Reconnect {attempt}/{self.max_reconnect_attempts}...",
                end=" ",
                flush=True,
            )
            logger.info(
                f"RTSP reconnect attempt {attempt}/{self.max_reconnect_attempts} — "
                f"waiting {self.reconnect_delay_sec}s."
            )
            time.sleep(self.reconnect_delay_sec)

            if self._open_source():
                print("SUCCESS")
                logger.info(f"RTSP stream reconnected on attempt {attempt}.")
                return True

            print("FAILED")

        logger.error(
            f"RTSP reconnection failed after {self.max_reconnect_attempts} attempt(s) "
            f"for '{display_source}'. Stream will be terminated."
        )
        print(
            f"\n  ERROR: Could not reconnect to RTSP camera after "
            f"{self.max_reconnect_attempts} attempt(s). Stopping stream.\n"
        )
        self._connected = False
        return False
