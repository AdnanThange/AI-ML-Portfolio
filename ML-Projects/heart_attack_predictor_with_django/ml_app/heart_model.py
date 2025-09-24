import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import joblib
import os

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths for saved objects
MODEL_FILE = os.path.join(BASE_DIR, "heart_model.pkl")
SCALER_FILE = os.path.join(BASE_DIR, "scaler.pkl")
PCA_FILE = os.path.join(BASE_DIR, "pca.pkl")
ENCODERS_FILE = os.path.join(BASE_DIR, "label_encoders.pkl")

# ---------------- Train model once ----------------
if not os.path.exists(MODEL_FILE):
    # Load dataset
    df = pd.read_csv(os.path.join(BASE_DIR, "..", "heart.csv"))

    # Encode categorical columns
    label_encoders = {}
    for col in ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Features and target
    X = df.drop(columns="HeartDisease")
    y = df["HeartDisease"]

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, train_size=0.8, random_state=42)

    # Train model
    model = RandomForestClassifier(max_depth=None, min_samples_leaf=10, min_samples_split=10)
    model.fit(X_train, y_train)

    # Save objects for prediction
    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    joblib.dump(pca, PCA_FILE)
    joblib.dump(label_encoders, ENCODERS_FILE)


# ---------------- Prediction function ----------------
def predict_heart_disease(input_data: dict):
    # Load saved objects
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    pca = joblib.load(PCA_FILE)
    label_encoders = joblib.load(ENCODERS_FILE)

    df = pd.DataFrame([input_data])

    # Encode categorical columns
    for col, le in label_encoders.items():
        if col in df.columns:
            df[col] = le.transform(df[col])

    # Scale and apply PCA
    X_scaled = scaler.transform(df)
    X_pca = pca.transform(X_scaled)

    # Predict
    pred = model.predict(X_pca)[0]

    return "Disease" if pred == 1 else "No Disease"
