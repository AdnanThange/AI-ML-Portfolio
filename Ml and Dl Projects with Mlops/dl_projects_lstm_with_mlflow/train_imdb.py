import os
import sys
import numpy as np
import mlflow
import mlflow.tensorflow
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score
from mlflow.models.signature import infer_signature

# -------------------------------
# 🔹 MLflow / Databricks setup
# -------------------------------
os.environ["DATABRICKS_HOST"] = "https://dbc-24fe64fc-5231.cloud.databricks.com"
os.environ["DATABRICKS_TOKEN"] = "dapi525ed5b453b93e8a21477ba74d8314f6"

mlflow.set_tracking_uri("databricks")

# Use a safe path under your user folder
mlflow.set_experiment("/Users/thangeadnan31@gmail.com/IMDB_LSTM_Run")

# -------------------------------
# 🔹 Dataset
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
# 🔹 Training & MLflow logging
# -------------------------------
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

    # Log parameters
    mlflow.log_param("max_features", max_features)
    mlflow.log_param("maxlen", maxlen)

    # Log metrics
    mlflow.log_metric("val_accuracy", history.history["val_accuracy"][-1])
    mlflow.log_metric("test_accuracy", test_acc)

    # Log model with signature
    example_input = x_train[:5]
    example_output = model.predict(example_input)
    signature = infer_signature(example_input, example_output)
    mlflow.tensorflow.log_model(model, "model", signature=signature, input_example=example_input)

    # Log predictions
    pred_file = "predictions.npy"
    np.save(pred_file, y_pred[:100])
    mlflow.log_artifact(pred_file, artifact_path="predictions")

    # Log training script
    script_path = os.path.abspath(__file__)
    if os.path.exists(script_path):
        mlflow.log_artifact(script_path, artifact_path="source_code")

    print(f"Run ID: {run.info.run_id}")
    print(f"Test Accuracy: {test_acc:.4f}")
