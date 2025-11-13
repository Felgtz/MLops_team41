import argparse, json
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error
import yaml
import mlflow

MODELS = {
    "linear": LinearRegression,
    "ridge": Ridge,
    "lasso": Lasso,
}

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--model_out", required=True)
    ap.add_argument("--metrics_out", required=True)
    return ap.parse_args()

def load_params():
    with open("../../../params.yaml", "r") as f:  # relative to wdir
        p = yaml.safe_load(f)["train_eval"]
    return p

def main():
    args = parse_args()
    params = load_params()
    model_name = params.get("model", "linear")
    random_state = int(params.get("random_state", 42))
    test_size = float(params.get("test_size", 0.2))

    # Data
    df = pd.read_csv(args.in_path)
    y = df["shares"]
    X = df.drop(columns=["shares"])

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # MLflow local tracking
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment("news_popularity")

    with mlflow.start_run():
        ModelCls = MODELS[model_name]
        model = ModelCls()
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)

        r2 = float(r2_score(y_te, preds))
        rmse = float(mean_squared_error(y_te, preds, squared=False))

        # Log to MLflow
        mlflow.log_params(
            {"model": model_name, "random_state": random_state, "test_size": test_size}
        )
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("rmse", rmse)
        mlflow.sklearn.log_model(model, artifact_path="model")

        # Persist artifacts
        joblib.dump(model, args.model_out)
        with open(args.metrics_out, "w") as f:
            json.dump({"r2": r2, "rmse": rmse}, f, indent=2)

if __name__ == "__main__":
    main()
