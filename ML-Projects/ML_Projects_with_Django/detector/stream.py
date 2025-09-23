import cv2
import torch
import time

model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
model.eval()

camera = cv2.VideoCapture(0)

def gen():
    while True:
        success, frame = camera.read()
        if not success:
            break

        frame = cv2.resize(frame, (640, 480))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=True):
                results = model(rgb_frame)

        annotated_frame = results.render()[0]

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        time.sleep(0.03)
