import os
import time
import cv2 as cv
from pseyepy import Camera

# 5-second delay before starting the capture
print("Starting in 5 seconds...")
time.sleep(5)

# Initialize the cameras
print("about to init")
cameras = Camera([0, 1, 2, 3], fps=60, resolution=Camera.RES_LARGE, colour=True)

# Define output directories for the images
output_dir = 'output_images'
os.makedirs(output_dir, exist_ok=True)
camera_dirs = [os.path.join(output_dir, f'camera_{i}') for i in range(4)]
for dir in camera_dirs:
    os.makedirs(dir, exist_ok=True)

# Define the duration of the recording (in seconds)
recording_duration = 5  # record for _ seconds
fps = 60  # frames per second
num_frames = recording_duration * fps

try:
    print("Capturing images...")
    for frame_idx in range(num_frames):
        frames, _ = cameras.read()
        for i, frame in enumerate(frames):
            file_name = os.path.join(camera_dirs[i], f'frame_{frame_idx:04d}.jpg')
            cv.imwrite(file_name, frame)  # Use OpenCV to save the image
        time.sleep(1/fps)
finally:
    print("Capture ended.")

# Close the camera resources
cameras.end()
