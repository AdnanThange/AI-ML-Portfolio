import cv2
from ultralytics import YOLO



model = YOLO("yolov8n.pt")  


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0] 
    annotated_frame = results.plot()  

    labels = results.names
    detected = [labels[int(cls)] for cls in results.boxes.cls]

    # Determine helmet status
    if "person" in detected and "helmet" in detected:
        text, color = "Helmet Detected", (0, 255, 0)
    elif "person" in detected:
        text, color = "No Helmet Detected", (0, 0, 255)
    else:
        text, color = "", None

    if text:
        cv2.putText(
            annotated_frame, text, (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3
        )
    cv2.imshow("Helmet Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
