# src/utils/io.py
"""
Light-weight I/O helpers used across the codebase.

Why keep a separate file?
-------------------------
• Avoid sprinkling `pathlib.Path.mkdir(..., exist_ok=True)` all over the place.  
• Centralise a couple of JSON helpers so every module serialises data the
  same way (UTF-8, pretty-printed, trailing newline, etc.).
• Keep the interface tiny on purpose—add new helpers only when they reach
  three call-sites or more.

Public API
----------
ensure_dir(path)               → Path         # mkdir -p and return the Path
save_json(obj, path)           → Path         # atomic, pretty
load_json(path, **kwargs)      → Any          # thin wrapper around json.load
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# --------------------------------------------------------------------------- #
# 1. Directory helper                                                         #
# --------------------------------------------------------------------------- #
def ensure_dir(path: str | Path) -> Path:
    """
    Create directory (and parents) if it doesn’t exist; return the Path.

    If you pass a *file* Path (e.g. ".../foo/bar.json"), the parent
    directory will be created.

    Examples
    --------
    >>> ensure_dir("artifacts/metrics")          # dir
    >>> ensure_dir("artifacts/models/xgb.pkl")   # file path, only parents made
    """
    p = Path(path)
    dir_path = p if p.suffix == "" else p.parent
    dir_path.mkdir(parents=True, exist_ok=True)
    return p


# --------------------------------------------------------------------------- #
# 2. JSON helpers (pretty & atomic)                                           #
# --------------------------------------------------------------------------- #
def _atomic_write(text: str, target: Path) -> None:
    """Write text to `target` using a temp file + replace to avoid corruption."""
    target = Path(target)
    ensure_dir(target.parent)

    with tempfile.NamedTemporaryFile("w", delete=False, dir=target.parent) as tmp:
        tmp.write(text)
        tmp.flush()
        tmp_path = Path(tmp.name)

    tmp_path.replace(target)  # atomic on the same filesystem
    logger.debug("Atomically wrote %s", target)


def save_json(obj: Any, path: str | Path) -> Path:
    """
    Serialise `obj` as UTF-8 JSON with 2-space indentation
    using an atomic write; returns the Path written.
    """
    path = Path(path)
    txt = json.dumps(obj, indent=2, ensure_ascii=False) + "\n"
    _atomic_write(txt, path)
    return path


def load_json(path: str | Path, **kwargs) -> Any:
    """
    Load JSON file and return the deserialised Python object.

    Extra keyword arguments are forwarded to json.load (e.g. object_pairs_hook).
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f, **kwargs)
    return data
