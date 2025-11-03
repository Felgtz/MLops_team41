#!/usr/bin/env python
"""
scripts/predict.py
==================

Batch-inference helper that delegates the heavy lifting to
`src.models.predict`.

Typical usage
-------------
# Score a CSV of new observations and write a CSV of predictions
python scripts/predict.py \
    --model artifacts/models/xgboost.pkl \
    --input data/processed/scoring_batch.csv \
    --features artifacts/feature_list.csv \
    --output scoring_batch_preds.csv

# If your model was trained on the original target scale
python scripts/predict.py ... --transform none
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.models.predict import run_prediction
from src.utils import ensure_dir

# --------------------------------------------------------------------------- #
# CLI arguments                                                               #
# --------------------------------------------------------------------------- #
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate predictions with a saved model.")
    p.add_argument(
        "--model",
        "-m",
        required=True,
        type=str,
        help="Path to the trained model (.pkl) in artifacts/models/",
    )
    p.add_argument(
        "--input",
        "-i",
        required=True,
        type=str,
        help="Path to the CSV file containing rows to score.",
    )
    p.add_argument(
        "--features",
        "-f",
        required=True,
        type=str,
        help="CSV with a single column 'feature' listing the training-time columns.",
    )
    p.add_argument(
        "--output",
        "-o",
        required=True,
        type=str,
        help="Where to save predictions (must end with .csv or .json).",
    )
    p.add_argument(
        "--transform",
        "-t",
        default="log1p",
        choices=["log1p", "none"],
        help="Target transform applied during training (default: log1p).",
    )
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Helper to persist predictions                                               #
# --------------------------------------------------------------------------- #
def _save_preds(preds, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    if out_path.suffix.lower() == ".csv":
        pd.Series(preds, name="prediction").to_csv(out_path, index=False)
    elif out_path.suffix.lower() == ".json":
        out_path.write_text(pd.Series(preds).to_json(orient="values", indent=2))
    else:
        raise ValueError("Output path must end with .csv or .json")
    logging.info("Saved %d predictions to %s", len(preds), out_path)


# --------------------------------------------------------------------------- #
# Main routine                                                                #
# --------------------------------------------------------------------------- #
def main() -> None:
    args = _parse_args()

    # Load input data and feature list
    df_in = pd.read_csv(args.input)
    feature_list = pd.read_csv(args.features)["feature"].tolist()

    preds = run_prediction(
        df=df_in,
        model_key=args.model,          # ← key, e.g. "xgboost"
        artifacts_dir="artifacts",     # adjust if you store models elsewhere
        feature_list=feature_list,
        transform=args.transform,
    )

    # Persist
    _save_preds(preds, Path(args.output))


# --------------------------------------------------------------------------- #
# Entry-point guard                                                           #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    main()
