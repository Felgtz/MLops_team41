## How to install on terminal
1. python -m venv mlopstec_env
2. set http_proxy=http://internet.ford.com:83
3. set https_proxy=http://internet.ford.com:83
4. set no_proxy=127.0.0.1,localhost,.ford.com
5. mlopstec_env\Scripts\activate (after running this line you will see mlopstec_env at the beginning of the command line)
6. pip install -e . --extra-index-url https://pypi.org/simple --extra-index-url https://pypi.ford.com/simple

## to run it on terminal
python -m online_news_popularity.pipeline.clean_dataset --raw       data/raw/online_news_modified.csv --reference data/raw/online_news_original.csv --final     data/processed/df_final_validated.csv



Online News Popularity — Data-Cleaning Pipeline
A reproducible, configurable Python library that cleans, validates, imputes and statistically calibrates the well-known “Online News Popularity” dataset (UCI Machine Learning Repository).

Key points

Pure Python ≥ 3.9, no compiled deps.
Modular: each step is a self-contained function (convert_all_numeric, enforce_logical_bounds, …), easy to unit-test or reuse on similar tabular data.
Configurable via Hydra YAML files or a minimal CLI fallback — works out-of-the-box even if Hydra isn’t installed.
Transparent: every function returns an audit report (counts of coerced / clipped / imputed cells), so you can log metrics to MLflow or CI.
Cloud-friendly I/O — local paths, s3://, gs://, az:// … everything goes through fsspec.
Table of Contents
Quick start
Project structure
Configuration
Developing / testing
Extending the pipeline
License
Quick start
1 · Install
bash

# clone & enter the repo
git clone https://github.com/your-org/online_news_popularity.git
cd online_news_popularity

# create a fresh env
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip

# install *editable* plus dev extras
pip install -e ".[dev,viz,hydra]"
(Or simply pip install -r requirements.txt if you don’t need editable mode.)

2 · Run cleaning pipeline
bash

# with Hydra (preferred)
onp-clean                                      \
    +paths.raw=data/raw/online_news_modified.csv \
    calibration.clip_k=2                        \
    --log-level INFO

# without Hydra (fallback CLI)
onp-clean --raw data/raw/online_news_modified.csv \
          --reference data/raw/online_news_original.csv \
          --final data/processed/clean.csv \
          --log-level INFO
3 · Inspect results
Cleaned dataset saved to data/processed/clean.csv (or the path you chose).
Logs show how many cells were coerced / clipped / imputed per rule.
Use Pandas as usual:
python

import pandas as pd
df = pd.read_csv("data/processed/clean.csv")
df.head()
Project structure

online_news_popularity/
├── __init__.py               # version, logger, public re-exports
├── io.py                     # fsspec-aware read_df / save_df
├── cleaning/
│   ├── type_coercion.py      # step 1
│   ├── enforcement.py        # step 2
│   ├── imputation.py         # step 3
│   └── calibration.py        # step 4
├── features/
│   └── groups.py             # single source of truth for column families
└── pipeline/
    └── clean_dataset.py      # end-to-end orchestrator (Hydra + CLI)
configs/
├── paths.yaml                # default file locations (raw / ref / final)
└── clean.yaml                # hyper-parameters for imputation & calibration
pyproject.toml                # build + tooling config
requirements.txt              # all-in-one dependency list
Configuration
Hydra lets you override any field at runtime without editing files.

Example — run on a different raw CSV and tighten Z-score clipping:

bash

onp-clean \
    +paths.raw="/mnt/nas/v3/online_news_modified.csv" \
    calibration.clip_k=2
Environment variables work too (ONP_RAW, ONP_REF, ONP_FINAL) thanks to the ${oc.env:…} syntax in paths.yaml.

Developing / testing
Lint & format

bash

ruff check .
black .
Type-check

bash

mypy online_news_popularity
Unit tests

bash

pytest -q
Build distribution

bash

python -m build
Extending the pipeline
Add a new feature family

Declare column names in features/groups.py.
Import that list inside type_coercion, enforcement, etc. as needed.
Custom imputation method

Edit cleaning/imputation.py — the policy dict is already passed in; plug your algorithm and update the report dictionary.
Alternative calibration

Replace or augment _zclip in calibration.py with quantile mapping, Box-Cox, etc.
Because each module is pure and stateless, you can unit-test changes in isolation and plug them back into the orchestrator without side effects.

License
MIT — see LICENSE file for the full text.