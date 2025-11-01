"""
src.config
==========

Single source of truth for all filesystem paths and lightweight helpers.
"""

from __future__ import annotations
from pathlib import Path
from typing import Iterable

PACKAGE_DIR: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = PACKAGE_DIR.parent

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

RAW_DATA_PATH: Path = RAW_DATA_DIR / "df_final_validated.csv"
READY_DATA_PATH: Path = PROCESSED_DATA_DIR / "df_ready_for_modeling.csv"

REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"

# aliases for older code -----------------------------------------------------
FIG_DIR: Path = FIGURES_DIR     # keeps legacy imports working
RANDOM_STATE: int = 42          # global seed used across modules

MODELS_DIR: Path = PROJECT_ROOT / "models"

def ensure_dirs(*dirs: Iterable[Path | str]) -> None:
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)

# create the canonical folder tree at import time
ensure_dirs(RAW_DATA_DIR, PROCESSED_DATA_DIR, FIGURES_DIR, MODELS_DIR)

__all__: list[str] = [
    "PACKAGE_DIR",
    "PROJECT_ROOT",
    "DATA_DIR",
    "RAW_DATA_DIR",
    "PROCESSED_DATA_DIR",
    "RAW_DATA_PATH",
    "READY_DATA_PATH",
    "REPORTS_DIR",
    "FIGURES_DIR",
    "FIG_DIR",          # new
    "MODELS_DIR",
    "RANDOM_STATE",     # new
    "ensure_dirs",
]
