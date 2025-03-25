import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
from sklearn.utils import shuffle

# Load Gesture Data
gesture_files = os.listdir("gesture_data")
gesture_names = [file.replace(".npy", "") for file in gesture_files]

X, y = [], []
for idx, file in enumerate(gesture_files):
    data = np.load(f"gesture_data/{file}")

    # Data Augmentation: Slight random noise to prevent overfitting
    noise = np.random.normal(0, 0.01, data.shape)
    data_augmented = data + noise

    X.append(data)
    X.append(data_augmented)  
    y.extend([idx] * len(data))
    y.extend([idx] * len(data_augmented))

# Convert to NumPy Arrays
X = np.array(X, dtype=object)
X = np.concatenate(X, axis=0)
y = np.array(y)

# Shuffle Data
X, y = shuffle(X, y, random_state=42)

# Define Model Architecture (Reduced Complexity)
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(X.shape[1],)),
    keras.layers.Dropout(0.3),  # Prevent Overfitting
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(len(gesture_names), activation='softmax')
])

# Compile Model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=30, validation_split=0.3, batch_size=32)  # Adjusted validation split

# Save Model
os.makedirs("model", exist_ok=True)
model.save("model/gesture_model.keras")  
print("✅ Model Trained and Saved!")
