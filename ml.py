import pandas as pd
import numpy as np
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import logging
import os
import joblib
from pathlib import Path
import json

# ===========================
# 1. Load Configuration from YAML
# ===========================
def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

# ===========================
# 2. Setup Logging
# ===========================
def setup_logging(log_path, level=logging.DEBUG):
    if not os.path.exists(os.path.dirname(log_path)):
        os.makedirs(os.path.dirname(log_path))
    
    logging.basicConfig(
        filename=log_path,
        level=level,
        format="%(asctime)s:%(levelname)s:%(message)s"
    )
    logger = logging.getLogger()
    return logger

# ===========================
# 3. Data Ingestion
# ===========================
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        logger.debug("Data loaded successfully")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None

# ===========================
# 4. Data Preprocessing
# ===========================
def preprocess_data(df, target_column):
    try:
        # Drop rows with missing values
        df = df.dropna()
        logger.debug("Missing values dropped")

        # Features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        logger.debug("Data split into train and test sets")

        # Scaling features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        logger.debug("Feature scaling done")

        return X_train_scaled, X_test_scaled, y_train, y_test
    except Exception as e:
        logger.error(f"Error preprocessing data: {e}")
        return None, None, None, None

# ===========================
# 5. Model Training
# ===========================
def train_model(X_train, y_train, model_save_path):
    try:
        model = LinearRegression()
        model.fit(X_train, y_train)
        logger.debug("Linear Regression model trained")

        # Ensure models folder exists
        model_dir = Path(model_save_path).parent
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save the trained model
        joblib.dump(model, model_save_path)
        logger.debug(f"Model saved at {model_save_path}")

        return model
    except Exception as e:
        logger.error(f"Error training model: {e}")
        return None

# ===========================
# 6. Model Evaluation
# ===========================
def evaluate_model(model, X_test, y_test, metrics_save_path):
    try:
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Log metrics
        logger.debug(f"Model evaluation done: MSE={mse}, R2={r2}")
        print(f"Mean Squared Error: {mse}")
        print(f"R2 Score: {r2}")

        # Convert metrics to JSON
        metrics = {
            "Mean_Squared_Error": mse,
            "R2_Score": r2
        }
        metrics_json = json.dumps(metrics, indent=4)

        # Save JSON to a file using pathlib
        output_path = Path(metrics_save_path).parent
        output_path.mkdir(parents=True, exist_ok=True)
        file_path = Path(metrics_save_path)
        file_path.write_text(metrics_json)

        logger.debug(f"Evaluation metrics saved to {file_path}")

    except Exception as e:
        logger.error(f"Error evaluating model: {e}")

# ===========================
# 7. Pipeline Execution
# ===========================
if __name__ == "__main__":
    config = load_config("config.yaml")

    # Load configuration values from the YAML file
    data_path = config['data']['path']
    target = config['data']['target_column']
    log_path = config['logging']['log_path']
    logging_level = getattr(logging, config['logging']['level'])
    model_save_path = config['model']['save_path']
    metrics_save_path = config['evaluation']['output_path']

    # Set up logging
    logger = setup_logging(log_path, level=logging_level)

    # Run the pipeline
    df = load_data(data_path)
    if df is not None:
        X_train, X_test, y_train, y_test = preprocess_data(df, target)
        if X_train is not None:
            model = train_model(X_train, y_train, model_save_path)
            if model is not None:
                evaluate_model(model, X_test, y_test, metrics_save_path)
