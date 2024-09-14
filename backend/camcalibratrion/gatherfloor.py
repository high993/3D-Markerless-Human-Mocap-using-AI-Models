from pseyepy import Camera
import cv2  # OpenCV for image handling
import os
import time


# 5-second delay before starting the capture
print("Starting in 5 seconds...")
time.sleep(5)

# Base directory to save images
base_dir = r'C:\Users\Intel\Desktop\pythonproject5\backend\cam_calibration'

# Create directories for saving images for each camera under the `cam_calibration` directory
for i in range(4):
    os.makedirs(os.path.join(base_dir, f'setfloorcam_{i}'), exist_ok=True)

# Initialize four cameras with specific parameters
cams = Camera([0, 1, 2, 3], fps=30, resolution=Camera.RES_LARGE, colour=True)

# Create windows for displaying the camera feeds
for i in range(4):
    cv2.namedWindow(f"Camera {i}", cv2.WINDOW_NORMAL)

try:
    for i in range(30):  # Capture 30 images
        # Read frames from all cameras
        frames, timestamps = cams.read()

        for j, frame in enumerate(frames):
            # Display the frame for each camera
            cv2.imshow(f"Camera {j}", frame)

            # Save the frame to the correct directory under `cam_calibration`
            filename = os.path.join(base_dir, f'setfloorcam_{j}', f'image_{i}.jpg')
            cv2.imwrite(filename, frame)
            print("saved to folder")
        # Wait for a short period between captures
        time.sleep(0.5)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # When finished, close all cameras and windows
    cams.end()
    cv2.destroyAllWindows()
