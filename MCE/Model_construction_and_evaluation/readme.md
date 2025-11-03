How to run

train --config "C:\Users\FGUTIE63\Desktop\MLOps\Fase 2\MLops_team41-main\MLops_team41-main\MCE\Model_construction_and_evaluation\src\config\experiment.yaml"

predict --model best_model --input X_test.csv --features artifacts/feature_list.csv --output preds.csv --transform none



# 🏗️  Tabular ML Boilerplate

**One-command** end-to-end pipeline for tabular (regression or classification)
projects:

1. Load & split a simple CSV  
2. Perform light feature-engineering  
3. Train a baseline or boosted model  
4. Persist artefacts + metrics  
5. Run batch inference

> Built with plain `pandas` + `scikit-learn` and optionally `xgboost`.  
> Everything is unit-tested, type-hinted, and PEP 517-compliant.

---

## 📁 Directory layout
. ├── src/ │ ├── config/ │ │ └── experiment.yaml # declarative experiment file │ ├── data/ │ │ ├── init.py │ │ └── load_and_split.py │ ├── features/ │ │ ├── init.py │ │ ├── drop_columns.py │ │ └── drop_high_corr_features.py │ ├── models/ │ │ ├── init.py │ │ ├── baseline.py │ │ ├── boosting.py │ │ ├── train.py │ │ └── predict.py │ └── pipelines/ │ ├── init.py │ └── training_pipeline.py # (stub - extend as you like) ├── scripts/ │ ├── train.py # CLI: train --config … │ └── predict.py # CLI: predict --input … --model … ├── tests/ # pytest unit tests └── pyproject.toml



---

## 🚀 Quick-start

### 0) Clone & install

```bash
git clone https://github.com/your-org/your-repo.git
cd your-repo

# core deps
pip install -e .

# optional extra for XGBoost
pip install -e .[xgboost]
1) Prepare your data
Put a single CSV under data/raw/, e.g.


data/raw/housing.csv
The file must contain the target column you specify in the config (price in the example).

2) Configure the experiment
Edit src/config/experiment.yaml:

yaml

data:
  path: "data/raw/housing.csv"
  target_column: "price"
  test_size: 0.2
  random_state: 42
preprocessing:
  drop_columns: ["id", "zipcode"]
  drop_high_corr:
    threshold: 0.95
model:
  type: "xgboost"            # or "baseline"
  params:
    n_estimators: 500
training:
  cv_folds: 5
output:
  artifacts_dir: "artifacts"
3) Train
bash

python scripts/train.py --config src/config/experiment.yaml
Outputs:


artifacts/
├── models/
│   └── xgboost.pkl
└── metrics/
    └── xgboost_metrics.json
4) Predict
bash

python scripts/predict.py \
  --input data/raw/housing.csv \
  --output predictions.csv    \
  --model xgboost
🧰 Library API
python

from src.data import load_csv, train_test_split_df
from src.features import DropColumns, DropHighCorrFeatures
from src.models.train import train_model
from src.models.predict import run_prediction
Module	Key functions/classes
src.data	load_csv, train_test_split_df
src.features	DropColumns, DropHighCorrFeatures
src.models.baseline	BaselineConfig, train_and_evaluate()
src.models.boosting	BoostingConfig, train_and_evaluate()
src.models.train	train_model() – unified façade
src.models.predict	run_prediction(), list_available_models()
🧪 Testing & linting
bash

# run unit tests
pytest -q

# static analysis
ruff check .
mypy src/ tests/
✨ Extending
New feature transformer
Add src/features/my_transformer.py inheriting BaseEstimator + TransformerMixin, then export it in src/features/__init__.py.

New model back-end

Implement src/models/lightgbm.py with a dataclass LightGBMConfig and train_and_evaluate() routine.

Register it in src/models/train.py:

python

from .lightgbm import LightGBMConfig, train_and_evaluate as _run_lgbm
_REGISTRY["lightgbm"] = (_run_lgbm, LightGBMConfig)
More complex data sources
Swap load_and_split.py for a reader that pulls from SQL, BigQuery, or a feature store—the rest of the pipeline stays unchanged.

🤝 Contributing
PRs are very welcome! Please:

Open an issue to discuss major changes first.
Create a feature branch (git checkout -b feat/my-feature).
Ensure pytest, ruff, and mypy pass.
Submit a pull request describing why the change is needed.
📄 License
MIT — see LICENSE for details.


