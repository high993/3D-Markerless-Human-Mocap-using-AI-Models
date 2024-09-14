import json
import time
from flask import Flask, jsonify
from flask_socketio import SocketIO
from flask_cors import CORS
import subprocess
from flask import request

app = Flask(__name__)

# CORS configuration to handle preflight requests and allow access from your frontend
CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}}, supports_credentials=True)

socketio = SocketIO(app, cors_allowed_origins="http://localhost:5173")

# Global variable to hold the camera index
camera_index = 2

# Handle camera index change event from the frontend
@socketio.on('change_camera_index')
def handle_camera_index_change(data):
    global camera_index
    print("Camera index change event received:", data)  # Log when event is received
    camera_index = int(data['camera_index'])
    print(f"Camera index changed to: {camera_index}")  # Log the new camera index

# Gather intrinsic data using the updated camera index
@app.route('/run-gather-intrinsic', methods=['POST', 'OPTIONS'])
def run_gather_intrinsic():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'CORS preflight successful'}), 200
    
    global camera_index
    print(f"Gather intrinsic data script triggered for Camera {camera_index}")
    script_path = r'C:\Users\Intel\Desktop\pythonproject5\backend\camcalibratrion\gatherintrisic.py'
    python_executable = r'C:\Users\Intel\anaconda3\envs\pseyepy_env\python.exe'
    
    # Use subprocess.run to make Flask wait for the script to finish
    result = subprocess.run([python_executable, script_path, str(camera_index)], capture_output=True, text=True)
    print(result.stdout)  # Log the script output
    return jsonify({'message': f'Gather intrinsic script completed for camera {camera_index}'})

# Camera calibration route
@app.route('/run-calibrate-camera', methods=['POST', 'OPTIONS'])
def run_calibrate_camera():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'CORS preflight successful'}), 200
    
    global camera_index
    print(f"Camera calibration script triggered for Camera {camera_index}")
    script_path = r'C:\Users\Intel\Desktop\pythonproject5\backend\camcalibratrion\intrinsiccalc.py'
    python_executable = r'C:\Users\Intel\anaconda3\envs\pseyepy_env\python.exe'
    
    # Run the calibration script synchronously
    result = subprocess.run([python_executable, script_path, str(camera_index)], capture_output=True, text=True)
    print(result.stdout)  # Log the output from the script
    return jsonify({'message': f'Camera calibration script completed for camera {camera_index}'})

# Gather floor data
@app.route('/run-gatherfloor', methods=['POST', 'OPTIONS'])
def run_gatherfloor():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'CORS preflight successful'}), 200
    
    print("Gather floor script triggered")
    script_path = r'C:\Users\Intel\Desktop\pythonproject5\backend\camcalibratrion\gatherfloor.py'
    python_executable = r'C:\Users\Intel\anaconda3\envs\pseyepy_env\python.exe'
    subprocess.Popen([python_executable, script_path])
    return jsonify({'message': 'Gather floor script started'})

# Extrinsic calibration
@app.route('/run-extrinsic-calibration', methods=['POST', 'OPTIONS'])
def run_extrinsic_calibration():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'CORS preflight successful'}), 200

    print("Extrinsics Calculation Triggered")
    script_path = r'C:\Users\Intel\Desktop\pythonproject5\backend\camcalibratrion\extrinsic.py'
    python_executable = r'C:\Users\Intel\anaconda3\envs\pseyepy_env\python.exe'
    subprocess.Popen([python_executable, script_path])
    return jsonify({'message': 'Extrinsics Calculated'})

# Capture data
@app.route('/run-capture-data', methods=['POST', 'OPTIONS'])
def run_capture_data():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'CORS preflight successful'}), 200

    print("Extrinsics Calculated")
    script_path = r'C:\Users\Intel\Desktop\pythonproject5\backend\capture.py'
    python_executable = r'C:\Users\Intel\anaconda3\envs\pseyepy_env\python.exe'
    subprocess.Popen([python_executable, script_path])
    return jsonify({'message': 'Frames Captured'})

# Pose estimation
@app.route('/run-pose', methods=['POST', 'OPTIONS'])
def run_pose():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'CORS preflight successful'}), 200

    print("Extrinsics Calculated")
    script_path = r'C:\Users\Intel\Desktop\pythonproject5\backend\pose.py'
    python_executable = r'C:\Users\Intel\anaconda3\envs\pseyepy_env\python.exe'
    subprocess.Popen([python_executable, script_path])
    return jsonify({'message': '2D Points Collected'})

# 3D Triangulation
@app.route('/run-3D-Triangulation', methods=['POST', 'OPTIONS'])
def run_3D_Triangulation():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'CORS preflight successful'}), 200

    print("Extrinsics Refined")
    script_path = r'C:\Users\Intel\Desktop\pythonproject5\backend\3dcalc.py'
    python_executable = r'C:\Users\Intel\anaconda3\envs\pseyepy_env\python.exe'
    subprocess.Popen([python_executable, script_path])
    return jsonify({'message': '3D Points Calculated'})

@app.route('/run-playback', methods=['POST', 'OPTIONS'])
def run_playback():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'CORS preflight successful'}), 200

    print("Playback started")
    try:
        # Open and read the 3D points file
        with open('output_3d_points.json', 'r') as f:
            data = json.load(f)

        # Send each frame to the frontend one by one
        for frame_data in data:
            if isinstance(frame_data, list) and len(frame_data) == 33:  # Ensure each frame has 33 points
                print(f"Sending frame: {frame_data}")  # Add debug print to log each frame being sent
                socketio.emit('3d-points', frame_data)
                socketio.sleep(0.016)  # Delay to simulate real-time playback
            else:
                print(f"Invalid frame data: {frame_data}")
        return jsonify({'message': 'Playback completed'})

    except FileNotFoundError:
        return jsonify({'message': '3D points file not found'}), 500
    except json.JSONDecodeError:
        return jsonify({'message': 'Error decoding 3D points JSON file'}), 500


# Add data to Neural Net
@app.route('/run-add-data', methods=['POST', 'OPTIONS'])
def run_add_data():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'CORS preflight successful'}), 200

    print("Extrinsics Refined")
    script_path = r'C:\Users\Intel\Desktop\pythonproject5\backend\nural net.py'
    python_executable = r'C:\Users\Intel\anaconda3\envs\pseyepy_env\python.exe'
    subprocess.Popen([python_executable, script_path])
    return jsonify({'message': 'Points Added to Neural Net'})

# Main entry point for running the Flask-SocketIO server
if __name__ == '__main__':
    socketio.run(app, port=3001)
