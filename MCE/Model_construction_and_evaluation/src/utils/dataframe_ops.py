# src/utils/dataframe_ops.py
import pandas as pd
from typing import Iterable

def drop_columns(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    """
    Return a copy of *df* with the columns in *cols* removed.
    Silently ignores labels that aren't present.
    """
    return df.drop(columns=list(cols), errors="ignore")
