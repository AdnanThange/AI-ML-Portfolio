import pandas as pd
from pathlib import Path

def load_data(input_path="data/raw/data.csv", output_path="data/loaded/data.csv"):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_path)
    df.to_csv(output_path, index=False)
    print(f"Data loaded and saved to {output_path}")
    return df

# Optional: run as script
if __name__ == "__main__":
    load_data()
