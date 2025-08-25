# evaluation.py
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from pathlib import Path
import json

def evaluate_model():
    INPUT_FOLDER = "data/processed"
    MODEL_PATH = "models/model.joblib"
    OUTPUT_METRICS = "evaluation/metrics.json"

    Path(OUTPUT_METRICS).parent.mkdir(parents=True, exist_ok=True)

    X_test = pd.read_csv(f"{INPUT_FOLDER}/X_test.csv")
    y_test = pd.read_csv(f"{INPUT_FOLDER}/y_test.csv")
    model = joblib.load(MODEL_PATH)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics = {"Mean_Squared_Error": mse, "R2_Score": r2}
    with open(OUTPUT_METRICS, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Evaluation metrics saved to {OUTPUT_METRICS}")
    print(f"MSE: {mse}, R2: {r2}")

# Only run when called directly
if __name__ == "__main__":
    evaluate_model()
