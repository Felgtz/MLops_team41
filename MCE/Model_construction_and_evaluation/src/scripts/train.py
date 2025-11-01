#!/usr/bin/env python
"""
scripts/train.py
================

Command-line entry-point for running the full experiment defined in a YAML
configuration file.

Example
-------
# Default location (src/config/experiment.yaml)
python scripts/train.py

# Custom config path
python scripts/train.py --config path/to/another_experiment.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.pipelines.training_pipeline import TrainingPipeline


# --------------------------------------------------------------------------- #
# Argument parsing                                                            #
# --------------------------------------------------------------------------- #
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train baseline + boosting models according to a YAML config."
    )
    p.add_argument(
        "--config",
        "-c",
        type=str,
        default="src/config/experiment.yaml",
        help="Path to the experiment YAML file.",
    )
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Main routine                                                                #
# --------------------------------------------------------------------------- #
def main() -> None:
    args = _parse_args()
    cfg_path = Path(args.config)

    logging.info("Launching TrainingPipeline with config: %s", cfg_path)
    leaderboard = TrainingPipeline(cfg_path).run()

    print("\n=== Final Leaderboard ===")
    print(
        leaderboard[["model", "RMSE_test", "MAE_test", "R2_test"]]
        .to_string(index=False, justify="center")
    )


# --------------------------------------------------------------------------- #
# Entry-point guard                                                           #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s: %(message)s",
    )
    main()
