import camera
from flask import Flask, render_template, Response

from camera_status import CameraStatus
from mood import Mood
from recommendation import *

app = Flask(__name__)


@app.route('/video_feed')
def video_feed():
    s = CameraStatus()
    s.enable()
    return Response(camera.execute(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/get_list', methods=['GET'])
def get_list():
    s = CameraStatus()
    m = Mood()
    s.disable()
    r = recommend(m)
    result = recommend_songs(r)
    return {"result": result}


@app.route('/turn_on_camera', methods=['POST'])
def turn_on_camera():
    s = CameraStatus()
    s.enable()
    return {"result": True}


if __name__ == '__main__':
    app.run(host="127.0.0.1", debug=True)
