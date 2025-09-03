




# train_student_performance_mlflow.py

import os
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score

# -------------------------------
# MLflow setup
# -------------------------------
mlflow.set_tracking_uri("databricks")  # Use "databricks" if you have env variables set
experiment_path = "/Shared/Student-Performance"
mlflow.set_experiment(experiment_path)

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv("Student_Performance.csv")

# -------------------------------
# Preprocessing
# -------------------------------
le = LabelEncoder()
df["Extracurricular Activities"] = le.fit_transform(df["Extracurricular Activities"])

x = df.drop(columns="Performance Index")
y = df["Performance Index"]

sc = StandardScaler()
x_scaled = sc.fit_transform(x)

# -------------------------------
# PCA
# -------------------------------
pca = PCA()
x_pca = pca.fit_transform(x_scaled)  # fit on scaled data

# -------------------------------
# Train-test split
# -------------------------------
x_train, x_test, y_train, y_test = train_test_split(
    x_pca, y, train_size=0.8, random_state=42
)

# -------------------------------
# MLflow run
# -------------------------------
with mlflow.start_run():

    # Log preprocessing parameters
    mlflow.log_param("scaler", "StandardScaler")
    mlflow.log_param("pca_components", x_pca.shape[1])
    mlflow.log_param("train_size", 0.8)
    mlflow.log_param("random_state", 42)

    # -------------------------------
    # Model
    # -------------------------------
    reg = LinearRegression()
    reg.fit(x_train, y_train)

    # -------------------------------
    # Predictions & metrics
    # -------------------------------
    y_predicted = reg.predict(x_test)
    r2 = r2_score(y_test, y_predicted)
    mlflow.log_metric("r2_score", r2)

    # -------------------------------
    # Log model
    # -------------------------------
    mlflow.sklearn.log_model(reg, "linear_regression_model")

    # Optional: log preprocessor as artifact
    preprocessor_path = "scaler_pca.npz"
    np.savez(preprocessor_path, scaler_mean=sc.mean_, scaler_scale=sc.scale_, pca_components=pca.components_)
    mlflow.log_artifact(preprocessor_path, artifact_path="preprocessor")

    # Optional: log script itself
    script_path = os.path.abspath(__file__)
    mlflow.log_artifact(script_path, artifact_path="source_code")

print(f"R2 score on test set: {r2:.4f}")




