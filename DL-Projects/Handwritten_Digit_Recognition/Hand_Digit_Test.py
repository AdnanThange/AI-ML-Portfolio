import cv2
import numpy as np
from keras.models import load_model

model = load_model("digits.h5")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.rectangle(frame, (100, 100), (300, 300), (255, 0, 0), 2)
    roi = frame[100:300, 100:300]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 2)
    _, threshold = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY_INV)
    resized = cv2.resize(threshold, (28, 28))
    normalized = resized / 255.0
    reshaped = normalized.reshape(1, 28, 28, 1)

    prediction = model.predict(reshaped)
    digit = np.argmax(prediction)

    cv2.putText(frame, f"Prediction: {digit}", (100, 90), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 2)
    cv2.imshow("Digit Recognition", frame)
    cv2.imshow("Thresholded ROI", threshold)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
