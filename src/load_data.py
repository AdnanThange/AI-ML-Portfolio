import pandas as pd
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


def load_data(input_path="data/raw/data.csv", output_path="data/loaded/data.csv"):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_path)
    df.to_csv(output_path, index=False)
    print(f"Data loaded and saved to {output_path}")
    return df

# Optional: run as script
if __name__ == "__main__":
    load_data()
