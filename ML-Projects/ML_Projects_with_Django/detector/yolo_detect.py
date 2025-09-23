from ultralytics import YOLO
import os
import cv2

# Load your YOLO model once
model = YOLO("yolov5n.pt")  # or your trained model path

def detect_objects(image_path):
    # ----------------------
    # Handle spaces in filename and avoid duplicates
    # ----------------------
    folder = os.path.dirname(image_path)
    base = os.path.basename(image_path)
    base = base.replace(" ", "_")
    new_image_path = os.path.join(folder, base)

    if image_path != new_image_path:
        counter = 1
        final_path = new_image_path
        while os.path.exists(final_path):
            name, ext = os.path.splitext(base)
            final_path = os.path.join(folder, f"{name}_{counter}{ext}")
            counter += 1
        os.rename(image_path, final_path)
        image_path = final_path

    # ----------------------
    # Run YOLO detection
    # ----------------------
    results = model.predict(image_path)
    results = results[0]  # take first (only) result

    # ----------------------
    # Get plotted image
    # ----------------------
    im = results.plot()  # returns image as numpy array

    # ----------------------
    # Save detected image in MEDIA_ROOT
    # ----------------------
    output_name = "detected_" + os.path.basename(image_path)
    output_path = os.path.join(folder, output_name)
    cv2.imwrite(output_path, im)  # save manually

    return output_path
