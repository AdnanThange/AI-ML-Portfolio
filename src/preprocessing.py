import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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


INPUT_CSV = "data/loaded/data.csv"   # output from load_data
OUTPUT_FOLDER = "data/processed"
TARGET_COLUMN = "Price ($)"           # your target column

# Ensure output folder exists
Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

# Load data
df = pd.read_csv(INPUT_CSV)

# Preprocessing function
def preprocess_data(df, target_column):
    df = df.dropna()
    print("Missing values dropped")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print("Data split into train and test sets")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Feature scaling done")

    return X_train_scaled, X_test_scaled, y_train, y_test

# Run preprocessing
X_train_scaled, X_test_scaled, y_train, y_test = preprocess_data(df, TARGET_COLUMN)

# Save processed files
pd.DataFrame(X_train_scaled, columns=df.drop(columns=[TARGET_COLUMN]).columns).to_csv(f"{OUTPUT_FOLDER}/X_train.csv", index=False)
pd.DataFrame(X_test_scaled, columns=df.drop(columns=[TARGET_COLUMN]).columns).to_csv(f"{OUTPUT_FOLDER}/X_test.csv", index=False)
y_train.to_csv(f"{OUTPUT_FOLDER}/y_train.csv", index=False)
y_test.to_csv(f"{OUTPUT_FOLDER}/y_test.csv", index=False)

print(f"Preprocessing done. Processed files saved to {OUTPUT_FOLDER}")
