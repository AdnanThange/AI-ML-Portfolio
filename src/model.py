import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
from pathlib import Path

import yaml

with open("params.yaml") as f:
    params = yaml.safe_load(f)

RAW_DATA = params["data"]["raw_path"]
PROCESSED_DATA_FOLDER = params["data"]["processed_path"]
TARGET_COLUMN = params["data"]["target_column"]

TEST_SIZE = params["preprocessing"]["test_size"]
RANDOM_STATE = params["preprocessing"]["random_state"]
MODEL_PATH = params["training"]["model_path"]
METRICS_PATH = params["evaluation"]["metrics_path"]


# Paths
INPUT_FOLDER = "data/processed"
OUTPUT_MODEL = "models/model.joblib"

# Ensure output folder exists
Path(OUTPUT_MODEL).parent.mkdir(parents=True, exist_ok=True)

# Load processed training data
X_train = pd.read_csv(f"{INPUT_FOLDER}/X_train.csv")
y_train = pd.read_csv(f"{INPUT_FOLDER}/y_train.csv")

# Train model function
def train_model(X_train, y_train, model_save_path):
    model = LinearRegression()
    model.fit(X_train, y_train)
    joblib.dump(model, model_save_path)
    print(f"Model trained and saved to {model_save_path}")
    return model

# Call function
train_model(X_train, y_train, OUTPUT_MODEL)
