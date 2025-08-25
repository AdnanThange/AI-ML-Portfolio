# main.py
from load_data import load_data
from preprocessing import preprocess_data
from model import train_model
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
RAW_DATA = "data/raw/data.csv"
PROCESSED_DATA_FOLDER = "data/processed"
MODEL_PATH = "models/model.joblib"
TARGET_COLUMN = "Price ($)"

# Step 1: Load data
df = load_data(RAW_DATA)

# Step 2: Preprocess data
X_train_scaled, X_test_scaled, y_train, y_test = preprocess_data(df, TARGET_COLUMN)

# Save processed data for DVC/evaluation stage
Path(PROCESSED_DATA_FOLDER).mkdir(parents=True, exist_ok=True)
import pandas as pd
pd.DataFrame(X_train_scaled).to_csv(f"{PROCESSED_DATA_FOLDER}/X_train.csv", index=False)
pd.DataFrame(X_test_scaled).to_csv(f"{PROCESSED_DATA_FOLDER}/X_test.csv", index=False)
y_train.to_csv(f"{PROCESSED_DATA_FOLDER}/y_train.csv", index=False)
y_test.to_csv(f"{PROCESSED_DATA_FOLDER}/y_test.csv", index=False)

# Step 3: Train model
train_model(X_train_scaled, y_train, MODEL_PATH)

print("Main pipeline finished: data loaded, preprocessed, model trained.")

from evaluation import evaluate_model

# At the end of main.py, after training:
evaluate_model()
