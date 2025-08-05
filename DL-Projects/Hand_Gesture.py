import cv2
import mediapipe as mp
import os
import pyautogui

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

finger_tips_ids = [4, 8, 12, 16, 20]

def get_finger_states(hand_landmarks):
    fingers = []
    fingers.append(1 if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x else 0)
    for tip_id in [8, 12, 16, 20]:
        fingers.append(1 if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[tip_id - 2].y else 0)
    return fingers

def detect_gesture(fingers):
    if fingers == [0, 0, 0, 0, 0]:
        return "Fist"
    elif fingers == [0, 1, 1, 0, 0]:
        return "Peace"
    elif fingers == [1, 0, 0, 0, 0]:
        return "Thumbs up"
    elif fingers == [1, 1, 1, 1, 1]:
        return "Open palm"
    else:
        return "Unknown"

def perform_action(gesture):
    if gesture == "Peace":
        os.system("start chrome")
    elif gesture == "Thumbs up":
        pyautogui.press("volumeup")
    elif gesture == "Fist":
        pyautogui.press("volumemute")
    elif gesture == "Open palm":
        os.system("explorer")

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
            fingers = get_finger_states(handLms)
            gesture = detect_gesture(fingers)
            perform_action(gesture)
            cv2.putText(frame, f'Gesture: {gesture}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("Hand Gesture Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
