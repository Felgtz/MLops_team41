# src/online_news_popularity/pipeline/clean_dataset.py
"""
End-to-end cleaning pipeline for the Online-News-Popularity dataset.

It can be launched in two ways:
1.  `onp-clean …`                     → Hydra mode (if hydra-core + configs/)
2.  `python -m online_news_popularity.pipeline.clean_dataset …`
    → fallback argparse CLI (no Hydra required)

Version 0.2.0 – adds URL normalisation & filtering
--------------------------------------------------
    • Strips / lowers the `url` column
    • Drops explicit bad tokens ["", "nan", "none", "null"]
    • Keeps only http/https URLs
    • De-duplicates on `url`, keeping the row with highest `shares`

The step runs right after reading the raw CSV, so all subsequent
cleaning stages see the filtered dataframe.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, Any

import pandas as pd

from ..cleaning import (
    type_coercion,
    enforcement,
    imputation,
    calibration,
)
from ..cleaning.url_filters import normalise_and_filter_urls

# ----------------------------------------------------------------------#
# Logging                                                               
# ----------------------------------------------------------------------#
LOG_FMT = "%(levelname)s | %(name)s | %(message)s"
log = logging.getLogger(__name__)
logging.basicConfig(
    level=os.getenv("PYTHONLOGGING", "INFO"),
    format=LOG_FMT,
)

# ----------------------------------------------------------------------#
# Core runner                                                           
# ----------------------------------------------------------------------#
def _run_pipeline(cfg: Dict[str, Any]) -> None:
    """
    Execute the full cleaning pipeline according to *cfg*.
    """
    paths = cfg["paths"]
    log.info("Reading raw dataset: %s", paths["raw"])
    df = pd.read_csv(paths["raw"])

    audit: Dict[str, Any] = {}

    # 1. URL normalisation, filtering & deduplication
    df, url_rpt = normalise_and_filter_urls(df, logger=log)
    audit["url_filter"] = url_rpt
    log.info(
        "Shape after URL filtering & deduplication: %s", df.shape
    )

    # 2. Type coercion (numeric columns only)
    df_tc, tc_rpt = type_coercion.convert_all_numeric(df, except_cols=["url"])
    audit["type_coercion"] = tc_rpt

    # 3. Enforcement of logical bounds
    df_enf, enf_rpt = enforcement.enforce_logical_bounds(df_tc, logger=log)
    audit["enforcement"] = enf_rpt

    # 4. Imputation
    df_imp, imp_rpt = imputation.impute(
        df_enf,
        policy=cfg.get("imputation", {}),
        logger=log,
    )
    audit["imputation"] = imp_rpt

    # 5. Calibration
    df_cal, cal_rpt = calibration.calibrate_to_reference(
        df_imp,
        paths["reference"],
        logger=log,
    )
    audit["calibration"] = cal_rpt

    # 6. Persist
    out_path = Path(paths["final"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_cal.to_csv(out_path, index=False)
    log.info("Saved cleaned dataset → %s  (shape=%s)", out_path, df_cal.shape)

    # (Optional) save audit report next to the dataframe
    (out_path.with_suffix(".audit.json")).write_text(
        pd.Series(audit).to_json(orient="table")
    )


# ----------------------------------------------------------------------#
# Fallback CLI (no Hydra)                                               
# ----------------------------------------------------------------------#
def _cli_fallback() -> None:
    """
    Simple argparse CLI mainly for CI pipelines or users without Hydra.
    """
    p = argparse.ArgumentParser(
        prog="clean_dataset",
        description="Clean the Online-News-Popularity dataset (no Hydra).",
    )
    p.add_argument("--raw", required=True, help="Path to raw CSV")
    p.add_argument("--reference", required=True, help="Reference CSV for calibration")
    p.add_argument(
        "--final",
        required=True,
        help="Destination path for the cleaned CSV",
    )
    p.add_argument(
        "--drop-row-na-pct",
        type=float,
        default=None,
        help="Optional: drop rows whose NaN percentage exceeds this value "
        "(applied *before* imputation)",
    )
    args = p.parse_args()

    cfg = {
        "paths": {
            "raw": args.raw,
            "reference": args.reference,
            "final": args.final,
        },
        "imputation": {
            "drop_rows_threshold_pct": args.drop_row_na_pct,
        },
        # calibration / other sections fall back to module defaults
    }
    _run_pipeline(cfg)


# ----------------------------------------------------------------------#
# Entry-point                                                           
# ----------------------------------------------------------------------#
def main() -> None:  # pragma: no cover
    """
    Decide whether to launch through Hydra or the fallback CLI.
    """
    try:
        import hydra  # noqa: F401

        from hydra import compose, initialize_config_dir

        root = Path(__file__).resolve().parents[3]  # repo root
        config_dir = root / "configs"
        if not config_dir.exists():
            raise FileNotFoundError

        initialize_config_dir(config_dir=str(config_dir))
        cfg = compose(config_name="clean")
        _run_pipeline(cfg.to_container(resolve=True))
    except Exception as err:  # any problem → fall back
        log.debug("Hydra boot failed (%s); falling back to argparse CLI.", err)
        _cli_fallback()


if __name__ == "__main__":
    main()
