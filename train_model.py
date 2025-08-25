import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from pathlib import Path
import json
# Data Loading




# ===========================
# Model Training
# ===========================
def train_model(X_train, y_train, model_save_path):
    try:
        model = LinearRegression()
        model.fit(X_train, y_train)
        print("Linear Regression model trained")

        Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_save_path)
        print(f"Model saved at {model_save_path}")

        return model
    except Exception as e:
        print(f"Error training model: {e}")
        return None