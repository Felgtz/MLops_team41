import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Any

BAD_TOKENS: List[str] = ["", "nan", "none", "null", None]

def normalise_and_filter_urls(df: pd.DataFrame,
                              shares_col: str = "shares",
                              logger: Any = None
                              ) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Clean the `url` column and drop unwanted rows.

    Returns
    -------
    df  : cleaned DataFrame (index reset)
    rpt : dict with counts of rows removed at each rule
    """
    rpt: Dict[str, int] = {}

    # Normalise: strip / lower
    df["url"] = df["url"].str.strip().str.lower()

    # 1. explicit bad tokens
    mask_bad = df["url"].isin(BAD_TOKENS)
    rpt["bad_tokens"] = int(mask_bad.sum())
    df = df.loc[~mask_bad]

    # 2. non-http/https
    mask_proto = ~df["url"].str.startswith(("http://", "https://"), na=False)
    rpt["non_http"] = int(mask_proto.sum())
    df = df.loc[~mask_proto]

    # 3. duplicates (keep highest shares)
    before = len(df)
    df = (
        df.sort_values(shares_col, ascending=False)
          .drop_duplicates(subset="url", keep="first")
          .reset_index(drop=True)
    )
    rpt["deduplicated"] = before - len(df)

    if logger:
        logger.info("URL filtering report: %s", rpt)
    return df, rpt