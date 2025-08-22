import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import load_model

DATASET_DIR = r"C:\Users\Adnan\OneDrive\Desktop\Brain_tumor_dataset\train"
BASE_DIR = "Human_Brain_Tumor"

def makedir():
    for folder in ['train/Yes', 'train/No', 'validation/Yes', 'validation/No']:
        path = os.path.join(BASE_DIR, folder)
        os.makedirs(path, exist_ok=True)

def split():
    yes_dir = os.path.join(DATASET_DIR, 'yes')
    no_dir = os.path.join(DATASET_DIR, 'no')

    yes_images = [img for img in os.listdir(yes_dir) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
    no_images = [img for img in os.listdir(no_dir) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]

    split_index_yes = int(len(yes_images) * 0.8)
    split_index_no = int(len(no_images) * 0.8)

    for i, img in enumerate(yes_images):
        src = os.path.join(yes_dir, img)
        if i < split_index_yes:
            dst = os.path.join(BASE_DIR, 'train', 'Yes', img)
        else:
            dst = os.path.join(BASE_DIR, 'validation', 'Yes', img)
        shutil.copy(src, dst)

    for i, img in enumerate(no_images):
        src = os.path.join(no_dir, img)
        if i < split_index_no:
            dst = os.path.join(BASE_DIR, 'train', 'No', img)
        else:
            dst = os.path.join(BASE_DIR, 'validation', 'No', img)
        shutil.copy(src, dst)

    print("Data split complete.")


def build_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(150, 150, 3)),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(300, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

def train(model):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        directory=os.path.join(BASE_DIR, 'train'),
        target_size=(150, 150),
        batch_size=32,
        class_mode="binary"
    )

    val_generator = val_datagen.flow_from_directory(
        directory=os.path.join(BASE_DIR, 'validation'),
        target_size=(150, 150),
        batch_size=32,
        class_mode="binary"
    )

    print(f"Train samples: {train_generator.samples}")
    print(f"Validation samples: {val_generator.samples}")

    model.fit(
        train_generator,
        epochs=10,
        steps_per_epoch=len(train_generator),
        validation_data=val_generator,
        validation_steps=len(val_generator)
    )

    model.save("brain_tumor_model.h5")
    print("Model saved as brain_tumor_model.h5")

def predict(model_path, image_path):
    model = load_model(model_path)
    image = load_img(image_path, target_size=(150, 150))
    img_array = img_to_array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0][0]
    result = "Yes" if prediction > 0.5 else "No"
    print(f"Tumor detected: {result} (Confidence: {prediction:.2f})")

if __name__ == "__main__":
    makedir()
    split()
    model = build_model()
    train(model)
    predict("brain_tumor_model.h5",r"C:\Users\Adnan\OneDrive\Desktop\hand gesture\Human_Brain_Tumor\validation\Yes\Y7.jpg")
