from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path
import json

def evaluate_model(model, X_test, y_test, metrics_save_path):
    try:
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"Mean Squared Error: {mse}")
        print(f"R2 Score: {r2}")

        metrics = {"Mean_Squared_Error": mse, "R2_Score": r2}
        metrics_json = json.dumps(metrics, indent=4)

        Path(metrics_save_path).parent.mkdir(parents=True, exist_ok=True)
        Path(metrics_save_path).write_text(metrics_json)
        print(f"Evaluation metrics saved to {metrics_save_path}")

    except Exception as e:
        print(f"Error evaluating model: {e}")
