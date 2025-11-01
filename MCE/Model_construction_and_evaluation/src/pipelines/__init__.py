"""
src.pipelines
=============

Namespace for end-to-end orchestration layers that tie data loading,
feature engineering, model training, and evaluation together.

Public API
----------
TrainingPipeline     – high-level helper that reads a YAML config and
                       coordinates the full experiment lifecycle.
"""

from __future__ import annotations

import logging

# --------------------------------------------------------------------------- #
# Re-exports                                                                  #
# --------------------------------------------------------------------------- #
from .training_pipeline import TrainingPipeline

__all__: list[str] = ["TrainingPipeline"]

# --------------------------------------------------------------------------- #
# Module-level logger (silenced by default)                                   #
# --------------------------------------------------------------------------- #
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
