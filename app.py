from flask import Flask, render_template, Response
import cv2
import pygame
import threading
import os

app = Flask(__name__)

# Initialize pygame mixer for sound
pygame.mixer.init()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Flag to keep track of whether the alarm is playing
is_alarm_playing = False

# Load the alarm sound
alarm_sound = "alarm.wav"  # Ensure you have an alarm.wav in the project directory

def play_alert():
    global is_alarm_playing
    if not is_alarm_playing:
        pygame.mixer.music.load(alarm_sound)
        pygame.mixer.music.play(-1)  # Play looped until stopped
        is_alarm_playing = True

def stop_alert():
    global is_alarm_playing
    if is_alarm_playing:
        pygame.mixer.music.stop()
        is_alarm_playing = False

def gen_frames():
    global is_alarm_playing
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray)

            if len(eyes) < 1:  # Eyes are closed
                play_alert()  # Start playing the alert sound
                cv2.putText(frame, "Eyes Closed", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            else:  # Eyes are open
                stop_alert()  # Stop the alert sound
                cv2.putText(frame, "Eyes Open", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
