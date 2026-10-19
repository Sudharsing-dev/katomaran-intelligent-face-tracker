"""Logging system for Katomaran Intelligent Face Tracker.

Provides structured file logging to logs/events.log and console output with
customizable formatting and severity levels.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

LOGGER_NAME = "katomaran"


def setup_logger(
    log_path: Optional[str | Path] = None,
    log_level: str = "INFO",
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB per log file
    backup_count: int = 5,
) -> logging.Logger:
    """Initialize and configure application-wide logger.

    Args:
        log_path: Destination path for events.log. Defaults to logs/events.log.
        log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL').
        max_bytes: Maximum size per log file before rotation.
        backup_count: Number of rotated backup logs to retain.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(LOGGER_NAME)

    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    # Avoid duplicate handlers if already configured
    if logger.handlers:
        return logger

    # Log format: Timestamp - Logger - Level - Message
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 1. Console Handler (stdout)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 2. Rotating File Handler (logs/events.log)
    if log_path:
        target_path = Path(log_path)
    else:
        target_path = Path(__file__).resolve().parent.parent / "logs" / "events.log"

    try:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            filename=str(target_path),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        logger.warning(f"Could not initialize file logging to '{target_path}': {e}")

    logger.propagate = False
    return logger


def get_logger() -> logging.Logger:
    """Retrieve existing application logger or create default if not yet initialized."""
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        return setup_logger()
    return logger
