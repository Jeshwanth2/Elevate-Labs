import cv2
import numpy as np
from flask import Flask, render_template, Response
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

# Load Model and Haar Cascade
# Check if model exists before loading
if not os.path.exists('models/mask_detector.h5'):
    print("Model not found! Creating a simple mock model for demonstration...")
    # Create a simple mock model for demonstration
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
    import numpy as np

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 1)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])

    # Save the mock model
    os.makedirs('models', exist_ok=True)
    model.save('models/mask_detector.h5')
    print("Mock model created and saved!")
else:
    model = load_model('models/mask_detector.h5')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

labels_dict = {0: 'MASK', 1: 'NO MASK'}
color_dict = {0: (0, 0, 255), 1: (0, 255, 0)} # Green for no mask, Red for mask

def generate_frames():
    camera = cv2.VideoCapture(0)
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                face_img = gray[y:y+h, x:x+w]
                resized = cv2.resize(face_img, (100, 100))
                normalized = resized / 255.0
                reshaped = np.reshape(normalized, (1, 100, 100, 1))
                
                # Predict
                result = model.predict(reshaped, verbose=0)
                label = np.argmax(result, axis=1)[0]
                
                # Draw Rectangle and Label
                cv2.rectangle(frame, (x, y), (x+w, y+h), color_dict[label], 2)
                cv2.rectangle(frame, (x, y-40), (x+w, y), color_dict[label], -1)
                
                # Alert Logic: If label == 1 (NO MASK), add alert text
                alert_text = labels_dict[label]
                if label == 1:
                    alert_text = "ALERT: NO MASK!"
                    # You can add system sound triggers here if desired

                cv2.putText(frame, alert_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Encode frame for Flask stream
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True, port=5000)