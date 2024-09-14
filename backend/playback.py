import json
import time
from flask_socketio import SocketIO
from flask import Flask

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

def playback_3d_points():
    # Load the 3D points data from the JSON file
    with open('output_3d_points.json', 'r') as file:
        points_data = json.load(file)

    # Emit the points frame by frame with a delay
    for frame in points_data:
        socketio.emit('3d-points', frame)
        time.sleep(0.016)  # 500ms delay between frames

if __name__ == '__main__':
    playback_3d_points()
