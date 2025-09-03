# train_student_performance.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv("Student_Performance.csv")

# -------------------------------
# Preprocessing
# -------------------------------
le = LabelEncoder()
df["Extracurricular Activities"] = le.fit_transform(df["Extracurricular Activities"])

X = df.drop(columns="Performance Index")
y = df["Performance Index"]

sc = StandardScaler()
X_scaled = sc.fit_transform(X)

# -------------------------------
# PCA
# -------------------------------
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# -------------------------------
# Train-test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, train_size=0.8, random_state=42
)

# -------------------------------
# Model training
# -------------------------------
reg = LinearRegression()
reg.fit(X_train, y_train)

# -------------------------------
# Predictions & metrics
# -------------------------------
y_pred = reg.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"R2 score on test set: {r2:.4f}")

# -------------------------------
# Save preprocessor and PCA
# -------------------------------
preprocessor_path = "scaler_pca.npz"
np.savez(
    preprocessor_path,
    scaler_mean=sc.mean_,
    scaler_scale=sc.scale_,
    pca_components=pca.components_
)

print(f"Preprocessor and PCA saved to {preprocessor_path}")

# -------------------------------
# Save model
# -------------------------------
import joblib
model_path = "linear_regression_model.pkl"
joblib.dump(reg, model_path)
print(f"Model saved to {model_path}")
