from data_loading import load_data
from preprocessing import preprocess_data
from train_model import train_model
from evaluation import evaluate_model

# Paths and parameters
data_path = "data/housing.csv"
target_column = "median_house_value"
model_save_path = "models/linear_regression_model.pkl"
metrics_save_path = "evaluation/metrics.json"

# Pipeline execution
df = load_data(data_path)
if df is not None:
    X_train, X_test, y_train, y_test = preprocess_data(df, target_column)
    if X_train is not None:
        model = train_model(X_train, y_train, model_save_path)
        if model is not None:
            evaluate_model(model, X_test, y_test, metrics_save_path)
