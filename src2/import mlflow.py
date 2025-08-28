

import mlflow
import mlflow.keras
import mlflow.tensorflow
import numpy as np
import os
import sys
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score

mlflow.set_experiment("IMDB_LSTM_Run")

# Dataset
max_features = 10000
maxlen = 200
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=max_features)
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

# Model
model = keras.Sequential([
    layers.Embedding(max_features, 128),
    layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    layers.Dense(1, activation="sigmoid")
])
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Training
with mlflow.start_run() as run:
    history = model.fit(
        x_train, y_train,
        batch_size=64,
        epochs=1,
        validation_split=0.2,
        verbose=1
    )

    # Predictions
    y_pred_prob = model.predict(x_test)
    y_pred = (y_pred_prob > 0.5).astype(int)
    test_acc = accuracy_score(y_test, y_pred)

    # Log params & metrics
    mlflow.log_param("max_features", max_features)
    mlflow.log_param("maxlen", maxlen)
    mlflow.log_metric("val_accuracy", history.history["val_accuracy"][-1])
    mlflow.log_metric("test_accuracy", test_acc)

    # Log model
    mlflow.keras.log_model(model, "model")

    # -------------------------------
    # 🔹 Log predictions file
    # -------------------------------
    pred_file = "predictions.npy"
    np.save(pred_file, y_pred[:100])
    mlflow.log_artifact(pred_file, artifact_path="predictions")

    # -------------------------------
    # 🔹 Log training script itself
    # -------------------------------
    script_path = os.path.abspath(sys.argv[0])
    if os.path.exists(script_path):
        mlflow.log_artifact(script_path, artifact_path="source_code")

    print(f"Run ID: {run.info.run_id}")
    print(f"Test Accuracy: {test_acc:.4f}")
