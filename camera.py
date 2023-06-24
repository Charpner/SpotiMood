import cv2
import numpy as np
import tensorflow as tf

from camera_status import CameraStatus
from mood import Mood


# Code for real time prediction #
def execute():
    class_names = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
    model = tf.keras.models.load_model('C:/Users/Cem/PycharmProjects/SpotiMood/model.h5', compile=False)
    video = cv2.VideoCapture(0)
    faceDetect = cv2.CascadeClassifier('C:/Users/Cem/PycharmProjects/SpotiMood/haarcascade_frontalface_default.xml')
    camera_status = CameraStatus()
    mood = Mood()

    while True:
        success, frame = video.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = faceDetect.detectMultiScale(gray, 1.3, 3)

            if camera_status.getstatus():
                for x, y, w, h in faces:
                    sub_face_img = gray[y: y + h, x: x + w]
                    resized = cv2.resize(sub_face_img, (48, 48))
                    normalize = resized / 255.0
                    reshaped = np.reshape(normalize, (1, 48, 48, 1))
                    result = model.predict(reshaped)
                    label = np.argmax(result, axis=1)[0]
                    print(label)
                    mood.update_name(label)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)
                    cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
                    cv2.putText(frame, class_names[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
