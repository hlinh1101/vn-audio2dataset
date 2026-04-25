"""Logging setup for vn-audio2dataset."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logger(
    name: str = "vn-audio2dataset",
    level: str = "INFO",
    log_file: str | Path | None = None,
) -> logging.Logger:
    """Create or update a practical console/file logger.

    The function is safe to call more than once. Existing handlers are replaced
    so CLI runs do not duplicate log lines during tests or repeated imports.
    """

    logger = logging.getLogger(name)
    logger.setLevel(_parse_level(level))
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file is not None:
        file_path = Path(log_file)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def _parse_level(level: str) -> int:
    normalized = level.strip().upper()
    parsed = logging.getLevelName(normalized)
    if isinstance(parsed, int):
        return parsed
    raise ValueError(f"Unsupported log level: {level}")
