"""Performance monitoring and telemetry for Katomaran Intelligent Face Tracker.

Measures per-frame pipeline latencies (detection, tracking, recognition),
FPS throughput, and host hardware resource utilization (CPU, RAM, GPU).
"""

from __future__ import annotations

import collections
import time
from typing import Any, Deque, Dict, Optional

try:
    import psutil
except ImportError:
    psutil = None  # type: ignore

try:
    import torch
except ImportError:
    torch = None  # type: ignore


class PerformanceMonitor:
    """Tracks latency metrics across computer vision pipeline stages and system metrics."""

    def __init__(self, window_size: int = 60):
        self.window_size = window_size

        # Moving windows for timing in seconds
        self._detection_times: Deque[float] = collections.deque(maxlen=window_size)
        self._tracking_times: Deque[float] = collections.deque(maxlen=window_size)
        self._recognition_times: Deque[float] = collections.deque(maxlen=window_size)
        self._frame_times: Deque[float] = collections.deque(maxlen=window_size)

        # Lifetime counters
        self.total_frames_processed: int = 0
        self.start_time: float = time.time()
        self._last_frame_timestamp: Optional[float] = None

        # Process monitor
        self._process = psutil.Process() if psutil else None

    def start_frame(self) -> None:
        """Mark start of a new frame processing cycle."""
        now = time.time()
        if self._last_frame_timestamp is not None:
            delta = now - self._last_frame_timestamp
            if delta > 0:
                self._frame_times.append(delta)
        self._last_frame_timestamp = now
        self.total_frames_processed += 1

    def record_detection(self, duration: float) -> None:
        """Record detection stage execution time in seconds."""
        self._detection_times.append(duration)

    def record_tracking(self, duration: float) -> None:
        """Record tracking stage execution time in seconds."""
        self._tracking_times.append(duration)

    def record_recognition(self, duration: float) -> None:
        """Record facial embedding/recognition stage execution time in seconds."""
        self._recognition_times.append(duration)

    @property
    def current_fps(self) -> float:
        """Calculate moving average FPS over recent window."""
        if not self._frame_times:
            return 0.0
        avg_dt = sum(self._frame_times) / len(self._frame_times)
        return (1.0 / avg_dt) if avg_dt > 0 else 0.0

    @property
    def overall_fps(self) -> float:
        """Calculate lifetime average FPS since initialization."""
        elapsed = time.time() - self.start_time
        return (self.total_frames_processed / elapsed) if elapsed > 0 else 0.0

    def get_stage_averages_ms(self) -> Dict[str, float]:
        """Return average execution latencies in milliseconds."""
        def avg_ms(times: Deque[float]) -> float:
            return (sum(times) / len(times) * 1000.0) if times else 0.0

        return {
            "detection_ms": avg_ms(self._detection_times),
            "tracking_ms": avg_ms(self._tracking_times),
            "recognition_ms": avg_ms(self._recognition_times),
        }

    def get_system_resources(self) -> Dict[str, Any]:
        """Fetch system CPU, memory, and GPU usage metrics."""
        cpu_percent = 0.0
        memory_mb = 0.0
        if self._process:
            try:
                cpu_percent = self._process.cpu_percent()
                memory_mb = self._process.memory_info().rss / (1024 * 1024)
            except Exception:
                pass

        gpu_info: Dict[str, Any] = {"available": False}
        if torch is not None and torch.cuda.is_available():
            try:
                gpu_info["available"] = True
                gpu_info["device_name"] = torch.cuda.get_device_name(0)
                mem_alloc = torch.cuda.memory_allocated(0) / (1024 * 1024)
                gpu_info["memory_allocated_mb"] = round(mem_alloc, 1)
            except Exception:
                pass

        return {
            "cpu_percent": round(cpu_percent, 1),
            "memory_mb": round(memory_mb, 1),
            "gpu": gpu_info,
        }

    def get_summary(self) -> Dict[str, Any]:
        """Generate comprehensive performance summary dictionary."""
        stages = self.get_stage_averages_ms()
        sys_res = self.get_system_resources()
        elapsed_sec = round(time.time() - self.start_time, 2)

        return {
            "total_frames": self.total_frames_processed,
            "elapsed_seconds": elapsed_sec,
            "moving_fps": round(self.current_fps, 2),
            "overall_fps": round(self.overall_fps, 2),
            "avg_detection_latency_ms": round(stages["detection_ms"], 2),
            "avg_tracking_latency_ms": round(stages["tracking_ms"], 2),
            "avg_recognition_latency_ms": round(stages["recognition_ms"], 2),
            "cpu_percent": sys_res["cpu_percent"],
            "memory_rss_mb": sys_res["memory_mb"],
            "gpu": sys_res["gpu"],
        }
