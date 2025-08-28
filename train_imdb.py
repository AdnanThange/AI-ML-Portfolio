import os
import numpy as np
import mlflow
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score
from mlflow.models.signature import infer_signature

# -------------------------------
# 🔹 DagsHub MLflow authentication
# -------------------------------
DAGSHUB_TOKEN = "a8f790c3a8780ceb38dbb49cbb87575769841e40"  # your token
os.environ["MLFLOW_TRACKING_USERNAME"] = "thangeadnan31"   # your DagsHub username
os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN

# MLflow tracking URI for DagsHub
mlflow.set_tracking_uri("https://dagshub.com/thangeadnan31/AiMLproject2.mlflow")

# Set experiment
mlflow.set_experiment("IMDB_LSTM_Run")

# -------------------------------
# 🔹 Data Preparation
# -------------------------------
max_features = 10000
maxlen = 200
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=max_features)
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

# -------------------------------
# 🔹 Model
# -------------------------------
model = keras.Sequential([
    layers.Embedding(max_features, 128),
    layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    layers.Dense(1, activation="sigmoid")
])
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# -------------------------------
# 🔹 MLflow Logging
# -------------------------------
with mlflow.start_run() as run:
    # Train the model
    history = model.fit(
        x_train, y_train,
        batch_size=64,
        epochs=1,
        validation_split=0.2,
        verbose=1
    )

    # Predictions and test accuracy
    y_pred_prob = model.predict(x_test)
    y_pred = (y_pred_prob > 0.5).astype(int)
    test_acc = accuracy_score(y_test, y_pred)

    # Log parameters
    mlflow.log_param("max_features", max_features)
    mlflow.log_param("maxlen", maxlen)

    # Log metrics
    mlflow.log_metric("val_accuracy", history.history["val_accuracy"][-1])
    mlflow.log_metric("test_accuracy", test_acc)

    # -------------------------------
    # 🔹 Save model locally and log as artifact
    # -------------------------------
    model_file = "lstm_model.h5"
    model.save(model_file)  # Save locally
    mlflow.log_artifact(model_file, artifact_path="model")  # Log to MLflow

    # Save predictions locally and log
    pred_file = "predictions.npy"
    np.save(pred_file, y_pred[:100])
    mlflow.log_artifact(pred_file, artifact_path="predictions")

    # Log source code
    script_path = os.path.abspath(__file__)
    if os.path.exists(script_path):
        mlflow.log_artifact(script_path, artifact_path="source_code")

    print(f"Run ID: {run.info.run_id}")
    print(f"Test Accuracy: {test_acc:.4f}")

print(f"✅ Run complete! View it on DagsHub MLflow: https://dagshub.com/thangeadnan31/AiMLproject2.mlflow")
