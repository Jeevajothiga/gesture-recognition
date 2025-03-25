import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pyautogui
import time
import os
from collections import deque

# Load trained AI model
model = tf.keras.models.load_model("model/gesture_model.keras")

# Initialize MediaPipe Hand Tracking
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.6, max_num_hands=1)

# Gesture Order (If Brightness Up/Down is flipped, swap them here)
GESTURES = ["Brightness Down", "Brightness Up", "Lock Screen", "Minimize Windows",
            "Open Task Manager", "Volume Down", "Volume Up", "Zoom In", "Zoom Out"]

# Open Webcam
cap = cv2.VideoCapture(0)

prev_time = 0
fps_limit = 8  # Lower FPS to reduce lag
gesture_buffer = deque(maxlen=4)  # Stores last 4 gestures for smoothing
last_gesture = None  # Stores the last executed gesture to prevent repeat errors

while cap.isOpened():
    current_time = time.time()
    elapsed_time = current_time - prev_time

    if elapsed_time < 1 / fps_limit:
        continue  

    prev_time = current_time  

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    gesture_name = "No Gesture"

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract Hand Landmarks
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
            landmarks = landmarks.flatten().reshape(1, -1)

            # Predict Gesture
            prediction = model.predict(landmarks, verbose=0)
            gesture_index = np.argmax(prediction)
            confidence = prediction[0][gesture_index]

            # Add to buffer for smoothing
            gesture_buffer.append(GESTURES[gesture_index])

            # Only trigger action if the same gesture appears 3 times consecutively
            if confidence > 0.9 and gesture_buffer.count(GESTURES[gesture_index]) >= 3:
                gesture_name = GESTURES[gesture_index]

                # Ensure the same action is not repeated multiple times
                if gesture_name != last_gesture:
                    last_gesture = gesture_name

                    # Perform Actions
                    if gesture_name == "Brightness Up":
                        os.system("powershell (Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightnessMethods).WmiSetBrightness(1,80)")
                    elif gesture_name == "Brightness Down":
                        os.system("powershell (Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightnessMethods).WmiSetBrightness(1,30)")
                    elif gesture_name == "Lock Screen":
                        os.system("rundll32.exe user32.dll,LockWorkStation")
                    elif gesture_name == "Minimize Windows":
                        pyautogui.hotkey("win", "d")
                    elif gesture_name == "Open Task Manager":
                        pyautogui.hotkey("ctrl", "shift", "esc")
                    elif gesture_name == "Volume Down":
                        pyautogui.press("volumedown")
                    elif gesture_name == "Volume Up":
                        pyautogui.press("volumeup")
                    elif gesture_name == "Zoom In":
                        pyautogui.hotkey("ctrl", "+")
                    elif gesture_name == "Zoom Out":
                        pyautogui.hotkey("ctrl", "-")

    # Display Gesture Name
    cv2.putText(frame, f"Gesture: {gesture_name}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Hand Gesture AI", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
