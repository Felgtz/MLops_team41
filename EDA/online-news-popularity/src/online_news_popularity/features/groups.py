"""
Feature families (single source of truth)
=========================================

Every cleaning / enforcement / imputation module imports **these**
constants so we never repeat brittle column lists across the code-base.

If you add or drop a column in the dataset, change it *here* and the
rest of the pipeline updates automatically.

Only small helper comprehensions are used so the file stays import-time
light and has **zero** runtime dependencies except the standard library.
"""

from __future__ import annotations

# ----------------------------------------------------------------------
# Fundamental families
# ----------------------------------------------------------------------

# 5 topic probabilities that should lie in [0, 1] and sum ≈ 1
LDA_COLS = [f"LDA_{i:02d}" for i in range(5)]

# Mashable data-channel one-hots (binary)
CHANNEL_COLS = [
    "data_channel_is_lifestyle",
    "data_channel_is_entertainment",
    "data_channel_is_bus",
    "data_channel_is_socmed",
    "data_channel_is_tech",
    "data_channel_is_world",
]

# Day-of-week one-hots (binary)
WEEKDAY_COLS = [
    "weekday_is_monday",
    "weekday_is_tuesday",
    "weekday_is_wednesday",
    "weekday_is_thursday",
    "weekday_is_friday",
    "weekday_is_saturday",
    "weekday_is_sunday",
]

# Convenience superset
BIN_COLS = CHANNEL_COLS + WEEKDAY_COLS + ["is_weekend"]

# ----------------------------------------------------------------------
# Keyword statistics (many NaNs, three have “negative = sentinel”)
# ----------------------------------------------------------------------
KW_COLS = [
    "kw_min_min",
    "kw_max_min",
    "kw_avg_min",
    "kw_min_max",
    "kw_max_max",
    "kw_avg_max",
    "kw_min_avg",
    "kw_max_avg",
    "kw_avg_avg",
    "kw_min_total",
    "kw_max_total",
    "kw_avg_total",
]

# In these three columns **negative numbers mean “missing”** rather than
# invalid — we treat them as NaN, not as an error.
SPECIAL_KW_NEG = {"kw_min_min", "kw_avg_min", "kw_min_avg"}

# ----------------------------------------------------------------------
# Sentiment / subjectivity / polarity
# ----------------------------------------------------------------------
RATE_COLS = [
    "global_rate_positive_words",
    "global_rate_negative_words",
    "rate_positive_words",
    "rate_negative_words",
]

SUBJECTIVITY_COLS = [
    "global_subjectivity",
    "title_subjectivity",
    "abs_title_subjectivity",
]

POS_POLARITY_COLS = [
    "avg_positive_polarity",
    "min_positive_polarity",
    "max_positive_polarity",
]

NEG_POLARITY_COLS = [
    "avg_negative_polarity",
    "min_negative_polarity",
    "max_negative_polarity",
]

ANY_POLARITY_COLS = [
    "global_sentiment_polarity",
    "title_sentiment_polarity",
    "abs_title_sentiment_polarity",
]

# ----------------------------------------------------------------------
# Self-reference and token / count metrics
# ----------------------------------------------------------------------
SELF_REFERENCE_COLS = [
    "self_reference_min_shares",
    "self_reference_max_shares",
    "self_reference_avg_sharess",
]

COUNT_COLS = [
    "n_tokens_title",
    "n_tokens_content",
    "n_unique_tokens",
    "n_non_stop_words",
    "n_non_stop_unique_tokens",
    "num_hrefs",
    "num_self_hrefs",
    "num_imgs",
    "num_videos",
    "num_keywords",
    "average_token_length",
]

# ----------------------------------------------------------------------
# Continuous “other” metrics
# ----------------------------------------------------------------------
# Anything numeric that is *not* in one of the explicit families above.
# You can extend this list if later phases need additional constraints.
CONTINUOUS_OTHER_COLS = [
    "shares",            # target
    "timedelta",         # days between article publication and crawl
]

# ----------------------------------------------------------------------
# Utility helpers (optional ─ not used by pipeline but handy in notebooks)
# ----------------------------------------------------------------------
ALL_KNOWN_COLS = (
    LDA_COLS
    + BIN_COLS
    + KW_COLS
    + RATE_COLS
    + SUBJECTIVITY_COLS
    + POS_POLARITY_COLS
    + NEG_POLARITY_COLS
    + ANY_POLARITY_COLS
    + SELF_REFERENCE_COLS
    + COUNT_COLS
    + CONTINUOUS_OTHER_COLS
)

NUMERIC_EXPECTED = [c for c in ALL_KNOWN_COLS if c != "url"]
