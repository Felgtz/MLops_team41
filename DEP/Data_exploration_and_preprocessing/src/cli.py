"""
src.cli
=======

Team-41 MLOps – command-line interface.

Examples
--------
# Run EDA + preprocessing
mlops41 preprocess

# Placeholder: model training
mlops41 train
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

# Relative imports keep both the linter and direct execution happy
from . import (  # noqa: WPS433  (allow import * from package)
    logger,
    load_dataset,
    save_dataset,
    quick_info,
    scale_features,
    plot_histograms,
    plot_corr_heatmap,
    RAW_DATA_PATH,
    READY_DATA_PATH,
    ensure_dirs,
)

# ─────────────────────────────────────────────────────────────────────────────
# Sub-command implementations
# ─────────────────────────────────────────────────────────────────────────────


def cmd_preprocess(_: argparse.Namespace) -> None:
    """Run EDA + feature scaling and save the processed dataset."""
    logger.info("=== Pre-processing stage ===")

    # Ensure required folders exist
    ensure_dirs(Path("data/processed"), Path("reports/figures"))

    # 1️⃣  Load raw data
    df_raw = load_dataset(RAW_DATA_PATH)
    quick_info(df_raw, "Raw dataset")

    # 2️⃣  Scale features (leave the target column untouched)
    df_ready = scale_features(df_raw, exclude=["shares"])

    # 3️⃣  Plots for the report
    plot_histograms(df_ready, save=True)
    plot_corr_heatmap(df_ready, save=True)

    # 4️⃣  Save artefact
    save_dataset(df_ready, READY_DATA_PATH)
    logger.info("Saved processed dataset → %s", READY_DATA_PATH)
    logger.info("Pre-processing finished ✅")


def cmd_train(args: argparse.Namespace) -> None:
    """Placeholder for a future training routine."""
    logger.warning("Model-training step not implemented yet.")
    logger.debug("Received args: %s", args)


# ─────────────────────────────────────────────────────────────────────────────
# CLI plumbing
# ─────────────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlops41",
        description="Team-41 MLOps pipeline CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # preprocess ────────────────────────────────────────────────────────────
    sp_pre = sub.add_parser(
        "preprocess",
        help="Run EDA + scaling and save processed dataset",
    )
    sp_pre.set_defaults(func=cmd_preprocess)

    # train (placeholder) ───────────────────────────────────────────────────
    sp_train = sub.add_parser(
        "train",
        help="Train predictive model (coming soon)",
    )
    sp_train.set_defaults(func=cmd_train)

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # Each sub-parser set its handler with set_defaults(func=…)
    handler: Callable[[argparse.Namespace], None] = args.func
    handler(args)


# Allow “python -m src.cli …” in addition to the console script
if __name__ == "__main__":
    main()
