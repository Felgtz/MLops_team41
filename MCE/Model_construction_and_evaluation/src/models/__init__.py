"""
src.models
==========

One-stop import hub for everything under `src/models/`.

Typical usage
-------------
>>> from src.models import (
...     BaselineConfig, BoostingConfig,
...     get_baseline_models, get_boosting_models,
...     train_baselines, train_boosting,
...     eval_regression, run_experiment, predict,
... )

The heavy lifting lives in the individual sub-modules (baseline.py,
boosting.py, train.py, predict.py, evaluate.py).  This file only re-exports
their public APIs so the rest of the codebase never needs to remember
deep module paths.
"""

from __future__ import annotations

import logging

# --------------------------------------------------------------------------- #
# 1. Baseline regressors                                                      #
# --------------------------------------------------------------------------- #
from .baseline import (
    BaselineConfig,
    get_models as get_baseline_models,
    train_and_evaluate as train_baselines,
)

# --------------------------------------------------------------------------- #
# 2. Gradient-boosting regressors                                             #
# --------------------------------------------------------------------------- #
from .boosting import (
    BoostingConfig,
    get_models as get_boosting_models,
    train_and_evaluate as train_boosting,
)

# --------------------------------------------------------------------------- #
# 3. Generic metric helper                                                    #
# --------------------------------------------------------------------------- #
from .evaluate import eval_regression

# --------------------------------------------------------------------------- #
# 4. Higher-level orchestration & inference (optional imports)                #
# --------------------------------------------------------------------------- #
# These are wrapped in try/except so importing src.models doesn’t
# pull in heavier dependencies unless you actually call them.
try:
    from .train import run_experiment  # script-level orchestrator
except Exception:  # pragma: no cover
    run_experiment = None  # type: ignore

try:
    from .predict import predict  # batch inference helper
except Exception:  # pragma: no cover
    predict = None  # type: ignore

# --------------------------------------------------------------------------- #
# 5. Clean public surface                                                     #
# --------------------------------------------------------------------------- #
__all__: list[str] = [
    # Baselines
    "BaselineConfig",
    "get_baseline_models",
    "train_baselines",
    # Boosting
    "BoostingConfig",
    "get_boosting_models",
    "train_boosting",
    # Metrics
    "eval_regression",
    # Optional high-level helpers
    "run_experiment",
    "predict",
]

# --------------------------------------------------------------------------- #
# 6. Module-level logger                                                      #
# --------------------------------------------------------------------------- #
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
