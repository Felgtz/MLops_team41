# MCE/Model_construction_and_evaluation/tests/unit/test_train_eval_unit.py

import pandas as pd
import yaml
from pathlib import Path

from src.train_eval import prepare_data, load_params


def test_prepare_data_drops_non_numeric_and_nans():
    df = pd.DataFrame(
        {
            "url": [
                "http://mashable.com/a",
                "http://mashable.com/b",
                None,
            ],
            "num_feature_1": [1.0, 2.0, 3.0],
            "num_feature_2": [10.0, None, 30.0],
            "shares": [100, 200, None],
        }
    )

    X, y = prepare_data(df, target_col="shares")

    assert "url" not in X.columns
    assert not X.isna().any().any()
    assert not y.isna().any()
    assert len(X) == 1
    assert len(y) == 1


def test_load_params_reads_yaml(tmp_path, monkeypatch):
    params_content = {
        "train_eval": {
            "model": "ridge",
            "random_state": 123,
            "test_size": 0.25,
        }
    }
    params_path = tmp_path / "params.yaml"
    params_path.write_text(yaml.dump(params_content), encoding="utf-8")

    monkeypatch.chdir(tmp_path)

    params = load_params()
    assert "train_eval" in params
    te = params["train_eval"]
    assert te["model"] == "ridge"
    assert te["random_state"] == 123
    assert te["test_size"] == 0.25
