# MCE/Model_construction_and_evaluation/src/train_eval.py
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import json
import mlflow

def main():
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment("news_popularity")

    df_train = pd.read_csv("data/processed/train.csv")
    df_test = pd.read_csv("data/processed/test.csv")

    X_train, y_train = df_train.drop(columns=["shares"]), df_train["shares"]
    X_test, y_test = df_test.drop(columns=["shares"]), df_test["shares"]

    with mlflow.start_run():
        model = LinearRegression()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        r2 = r2_score(y_test, preds)
        rmse = mean_squared_error(y_test, preds, squared=False)

        mlflow.log_metric("r2", r2)
        mlflow.log_metric("rmse", rmse)
        mlflow.sklearn.log_model(model, "model")

        metrics = {"r2": r2, "rmse": rmse}
        with open("metrics/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        joblib.dump(model, "models/model.pkl")

if __name__ == "__main__":
    main()
