import cv2
import mediapipe as mp
import numpy as np
import os
import time

# Define Gesture Labels (ONLY THE SELECTED GESTURES)
GESTURE_LABELS = ["Volume Up", "Volume Down", "Lock Screen", "Brightness Up", "Brightness Down",
                  "Open Task Manager", "Minimize Windows", "Zoom In", "Zoom Out"]

# **Fix Input Freezing Issue**
print("Available Gestures: ", GESTURE_LABELS)
while True:
    try:
        gesture_index = int(input(f"Enter gesture index (0-{len(GESTURE_LABELS) - 1}): "))
        if 0 <= gesture_index < len(GESTURE_LABELS):
            break
        else:
            print("❌ Invalid choice. Enter a number between 0 and", len(GESTURE_LABELS) - 1)
    except ValueError:
        print("❌ Invalid input. Enter a number.")

gesture_name = GESTURE_LABELS[gesture_index]
print(f"✅ Collecting data for: {gesture_name}")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.5)

# Create Directory for Storing Data
os.makedirs("gesture_data", exist_ok=True)

# Open Webcam
cap = cv2.VideoCapture(0)

data = []
sample_count = 0
prev_time = 0
fps_limit = 10  # **Limit FPS to 10 for smooth capture**

print("🟢 Show the gesture! Press 'q' to stop...")

while cap.isOpened():
    current_time = time.time()
    elapsed_time = current_time - prev_time

    # **Limit frame rate to prevent lag**
    if elapsed_time < 1 / fps_limit:
        continue
    prev_time = current_time

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract Hand Landmarks
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
            data.append(landmarks.flatten())  # Flatten data for training
            sample_count += 1

    # Show Sample Count on Screen
    cv2.putText(frame, f"Gesture: {gesture_name} | Samples: {sample_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    cv2.imshow("Collecting Gesture Data", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save Data as NumPy File
if sample_count > 0:
    data = np.array(data)
    np.save(f"gesture_data/{gesture_name}.npy", data)
    print(f"✅ Data collected for {gesture_name}: {data.shape}")
else:
    print("⚠ No data collected. Try again.")

print("✅ Collection complete! Run `train_model.py` to train AI.")
