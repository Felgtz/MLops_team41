"""
src.logger
==========

Central logging utility for the Team-41 pipeline.
Importing this module once anywhere in the project guarantees that:

1. Logging is configured (formatter, level, handlers).
2. A module-level `logger` instance is available for convenience.

Example
-------
from src.logger import logger

logger.info("Training started…")
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------#
# Configuration helpers                                                      #
# ---------------------------------------------------------------------------#


def _default_log_level() -> int:
    """Return INFO unless the environment asks for more verbosity."""
    import os

    # Users can override with:  set PYTHON_LOG_LEVEL=DEBUG
    level_str = os.getenv("PYTHON_LOG_LEVEL", "INFO").upper()
    return getattr(logging, level_str, logging.INFO)


def _build_formatter() -> logging.Formatter:
    """Build a simple, readable formatter with time, level and message."""
    return logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _configure_root_logger() -> None:
    """Set up root logger exactly once (idempotent)."""
    root = logging.getLogger()

    # If handlers already attached, assume another module configured logging.
    if root.handlers:
        return

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(_build_formatter())

    root.addHandler(handler)
    root.setLevel(_default_log_level())


# ---------------------------------------------------------------------------#
# Public objects                                                              #
# ---------------------------------------------------------------------------#

_configure_root_logger()

# Use a package-specific child logger (so messages can be filtered separately)
logger = logging.getLogger("mlops_team41")

# Optional helper: convenience function to create log files if required
def add_file_handler(file_path: str | Path, level: int | str = "INFO") -> None:
    """
    Append a FileHandler that writes logs to *file_path*.

    Parameters
    ----------
    file_path : Path-like
        Destination file (will be created along with parent folders if needed).
    level : str or int, default "INFO"
        Log level for the file handler.
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(file_path, encoding="utf-8")
    fh.setFormatter(_build_formatter())
    fh.setLevel(getattr(logging, str(level).upper(), logging.INFO))

    logging.getLogger().addHandler(fh)
    logger.debug("Added file handler → %s (level %s)", file_path, level)
