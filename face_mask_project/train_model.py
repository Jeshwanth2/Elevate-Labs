import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

# Configurations
DATA_DIR = 'dataset'
CATEGORIES = ['with_mask', 'without_mask']
IMG_SIZE = 100 # Resizing to 100x100

data = []
target = []

# 1. Load and Preprocess Data
print("Loading and preprocessing images...")
for category in CATEGORIES:
    folder_path = os.path.join(DATA_DIR, category)
    label = CATEGORIES.index(category)
    
    if not os.path.exists(folder_path):
        print(f"Warning: Folder {folder_path} not found. Please add dataset.")
        continue

    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        try:
            img = cv2.imread(img_path)
            # Convert to grayscale as requested
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
            # Resize
            resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
            data.append(resized)
            target.append(label)
        except Exception as e:
            pass

data = np.array(data) / 255.0 # Normalize
# Reshape for CNN (Samples, Height, Width, Channels)
data = np.reshape(data, (data.shape[0], IMG_SIZE, IMG_SIZE, 1))
target = np.array(target)

# Train/Test Split
train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=0.2, random_state=42)

# 2. Build the CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax') # 2 output classes: Mask / No Mask
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 3. Train and Save
os.makedirs('models', exist_ok=True)
checkpoint = ModelCheckpoint('models/mask_detector.h5', monitor='val_loss', save_best_only=True, mode='auto')

print("Training model...")
history = model.fit(train_data, train_target, epochs=10, validation_data=(test_data, test_target), callbacks=[checkpoint])

print("Training complete! Model saved to models/mask_detector.h5")