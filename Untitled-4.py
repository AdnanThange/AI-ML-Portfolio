import torch
import cv2

# Load YOLOv5 model (downloads automatically if not cached)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Open webcam (use 0 for default camera)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Run detection
    results = model(frame)
    detections = results.xyxy[0]  # x1, y1, x2, y2, conf, cls

    # Convert result to numpy array for drawing
    img = frame.copy()

    helmet_detected = False
    person_detected = False

    for *box, conf, cls in detections:
        class_id = int(cls)
        label = model.names[class_id]

        # Draw bounding box
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Check detections
        if label.lower() == "person":
            person_detected = True
        if "helmet" in label.lower():  # If using custom helmet model
            helmet_detected = True

    # Add overall status text
    if helmet_detected:
        cv2.putText(img, "Helmet Detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    else:
        cv2.putText(img, "No Helmet Detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    if person_detected:
        cv2.putText(img, "Person Detected", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)

    cv2.imshow("Helmet Detection", img)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
