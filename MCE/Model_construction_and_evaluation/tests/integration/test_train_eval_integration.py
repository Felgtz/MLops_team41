# MCE/Model_construction_and_evaluation/tests/integration/test_train_eval_integration.py

import json
import sys
from pathlib import Path

import pandas as pd
import yaml

from src import train_eval


def test_train_eval_end_to_end(tmp_path, monkeypatch):
    df = pd.DataFrame(
        {
            "num_feature_1": [1.0, 2.0, 3.0, 4.0],
            "num_feature_2": [10.0, 20.0, 30.0, 40.0],
            "url": [
                "http://mashable.com/a",
                "http://mashable.com/b",
                "http://mashable.com/c",
                "http://mashable.com/d",
            ],
            "shares": [100, 200, 300, 400],
        }
    )
    data_path = tmp_path / "df_ready_for_modeling.csv"
    df.to_csv(data_path, index=False)

    params_content = {
        "train_eval": {
            "model": "ridge",
            "random_state": 42,
            "test_size": 0.25,
        }
    }
    params_path = tmp_path / "params.yaml"
    params_path.write_text(yaml.dump(params_content), encoding="utf-8")

    model_out = tmp_path / "models" / "model.pkl"
    metrics_out = tmp_path / "metrics" / "metrics.json"
    model_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)

    monkeypatch.chdir(tmp_path)

    original_argv = sys.argv
    sys.argv = [
        "train_eval",
        "--in",
        str(data_path),
        "--model_out",
        str(model_out),
        "--metrics_out",
        str(metrics_out),
    ]
    try:
        train_eval.main()
    finally:
        sys.argv = original_argv

    assert model_out.exists()
    assert metrics_out.exists()

    metrics = json.loads(metrics_out.read_text(encoding="utf-8"))
    assert "r2" in metrics
    assert "rmse" in metrics
