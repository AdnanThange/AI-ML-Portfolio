
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from pathlib import Path
import json

def preprocess_data(df, target_column):
    try:
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
    except Exception as e:
        print(f"Error preprocessing data: {e}")
        return None, None, None, None