import os
import json
import yaml
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import joblib

with open("params.yaml") as f:
    params = yaml.safe_load(f)

PROCESSED_DATA_FOLDER = params["data"]["processed_path"]
MODEL_PATH = params["training"]["model_path"]
METRICS_PATH = params["evaluation"]["metrics_path"]

def evaluate_model():
    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)

    X_test = pd.read_csv(f"{PROCESSED_DATA_FOLDER}/X_test.csv")
    y_test = pd.read_csv(f"{PROCESSED_DATA_FOLDER}/y_test.csv")
    model = joblib.load(MODEL_PATH)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics = {"Mean_Squared_Error": mse, "R2_Score": r2}
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Evaluation metrics saved to {METRICS_PATH}")
    print(f"MSE: {mse}, R2: {r2}")

if __name__ == "__main__":
    evaluate_model()
