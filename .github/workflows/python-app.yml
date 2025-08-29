# train_imdb_simple.py

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score

# -------------------------------
# Dataset
# -------------------------------
max_features = 10000
maxlen = 200

(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=max_features)
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

# -------------------------------
# Model
# -------------------------------
model = keras.Sequential([
    layers.Embedding(max_features, 128),
    layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    layers.Dense(1, activation="sigmoid")
])
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# -------------------------------
# Training
# -------------------------------
history = model.fit(
    x_train, y_train,
    batch_size=64,
    epochs=1,
    validation_split=0.2,
    verbose=1
)

# -------------------------------
# Evaluation
# -------------------------------
y_pred_prob = model.predict(x_test)
y_pred = (y_pred_prob > 0.5).astype(int)
test_acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {test_acc:.4f}")
