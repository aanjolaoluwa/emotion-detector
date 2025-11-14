# app.py
from flask import Flask, render_template, request
import os
import sqlite3
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import json

# === FLASK SETUP ===
app = Flask(__name__, template_folder='Templates')  # CAPITAL T!
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# === LOAD MODEL & LABELS ===
print("Loading model and labels...")
model = load_model('face_emotionModel.h5')
with open('class_labels.json', 'r') as f:
    class_labels = json.load(f)
print("Model loaded successfully!")

# === DATABASE SETUP ===
DB_NAME = 'database.db'

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            image_path TEXT NOT NULL,
            emotion TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# === CHEERFUL MESSAGES ===
CHEER_MESSAGES = {
    'sad': "You are frowning. Why are you sad? Don’t worry — every storm passes. You’re stronger than you know!",
    'angry': "You look angry. Take a deep breath. Tomorrow is a new day!",
    'fear': "You seem scared. It’s okay — you’ve got this. One step at a time!",
    'disgust': "Hmm, something bothering you? Let it go — peace feels better!",
}

# === PREDICTION FUNCTION ===
def predict_emotion(image_path):
    img = load_img(image_path, color_mode='grayscale', target_size=(48, 48))
    img_array = img_to_array(img)
    img_array = img_array.reshape(1, 48, 48, 1).astype('float32') / 255.0
    prediction = model.predict(img_array, verbose=0)
    emotion_idx = np.argmax(prediction)
    emotion = class_labels[emotion_idx]
    confidence = float(np.max(prediction))
    return emotion, confidence

# === ROUTE ===
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        file = request.files['photo']

        if file and name and email:
            filename = f"{email.split('@')[0]}_{file.filename}"
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(image_path)

            emotion, confidence = predict_emotion(image_path)
            message = CHEER_MESSAGES.get(emotion, f"You are {emotion}. Great job!")

            # Save to DB
            conn = sqlite3.connect(DB_NAME)
            c = conn.cursor()
            c.execute("INSERT INTO users (name, email, image_path, emotion) VALUES (?, ?, ?, ?)",
                      (name, email, image_path, emotion))
            conn.commit()
            conn.close()

            return render_template('index.html',
                                   message=message,
                                   emotion=emotion.capitalize(),
                                   confidence=round(confidence * 100, 1),
                                   image_url=filename)

    return render_template('index.html')

# === RUN SERVER (RENDER COMPATIBLE) ===
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=False)
