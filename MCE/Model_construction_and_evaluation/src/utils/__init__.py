"""
src.utils
=========

Tiny helper namespace that re-exports the handful of generic I/O utilities
used across the project.  Keeping imports shallow means you can simply do

    from src.utils import ensure_dir, save_json

instead of remembering the deeper sub-module path.

Public API
----------
ensure_dir(path)        → Path
save_json(obj, path)    → Path
load_json(path, ...)    → Any
"""

from __future__ import annotations

import logging

# --------------------------------------------------------------------------- #
# Re-export from .io                                                          #
# --------------------------------------------------------------------------- #
from .io import ensure_dir, load_json, save_json

__all__: list[str] = [
    "ensure_dir",
    "save_json",
    "load_json",
]

# --------------------------------------------------------------------------- #
# Module-level logger                                                         #
# --------------------------------------------------------------------------- #
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
