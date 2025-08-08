import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import load_model

IMAGE_SIZE = 255
BATCH_SIZE = 32

TRAIN_DIR = r"C:\Users\Adnan\OneDrive\Desktop\hand gesture\Pneumoniz\chest_xray\train"
TEST_DIR = r"C:\Users\Adnan\OneDrive\Desktop\hand gesture\Pneumoniz\chest_xray\test"
VAL_DIR = r"C:\Users\Adnan\OneDrive\Desktop\hand gesture\Pneumoniz\chest_xray\val"

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    height_shift_range=0.2,
    width_shift_range=0.2
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

def build_model():
    base_model = VGG16(weights="imagenet", include_top=False,
                       input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    for layer in base_model.layers:
        layer.trainable = False
    model = models.Sequential([
        base_model,
        layers.Flatten(),
        layers.Dense(290, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid")
    ])
    return model

def predict_image(model_path, image_path):
    model = load_model(model_path)
    img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    print(prediction)

if __name__ == "__main__":
    model = build_model()
    model.summary()
    predict_image(
        "trained_network.h5",
        r"C:\Users\Adnan\OneDrive\Desktop\hand gesture\Pneumoniz\chest_xray\train\NORMAL\IM-0115-0001.jpeg"
    )
