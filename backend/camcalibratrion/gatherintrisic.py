import cv2
import time
from pseyepy import Camera
import os
import sys  # For getting the camera index from arguments

# Get camera index (i) from command line arguments
if len(sys.argv) > 1:
    i = int(sys.argv[1])  # Camera index passed as an argument
else:
    i = 2  # Default camera index

# 5-second delay before starting the capture
print("Starting in 7 seconds...")
time.sleep(7)

# Initialize both cameras
cams = Camera(ids=[0, 1, 2, 3], fps=30, resolution=Camera.RES_LARGE)

# Create window names for the cameras
window_name_cam0 = "Camera 0"
window_name_cam1 = "Camera 1"
window_name_cam2 = "Camera 2"
window_name_cam3 = "Camera 3"

cv2.namedWindow(window_name_cam0)
cv2.namedWindow(window_name_cam1)
cv2.namedWindow(window_name_cam2)
cv2.namedWindow(window_name_cam3)

# Absolute directory path to save images
output_dir = os.path.join(r'C:\Users\Intel\Desktop\pythonproject5\cam_calibration_intrinsic', f'intrinsiccam_{i}')
os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

# Counter for the number of images saved
image_count = 0
max_images = 30

try:
    while image_count < max_images:
        # Read frames from both cameras
        frames, timestamps = cams.read()

        # Display frames in separate windows
        cv2.imshow(window_name_cam0, frames[0])
        cv2.imshow(window_name_cam1, frames[1])
        cv2.imshow(window_name_cam2, frames[2])
        cv2.imshow(window_name_cam3, frames[3])

        # Save image from the selected camera (i)
        filename = os.path.join(output_dir, f'image_{image_count}.jpg')
        frame = cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR)
        if cv2.imwrite(filename, frame):
            print(f"Image saved as {filename}")
            image_count += 1
        else:
            print(f"Error: Could not save image {filename}")

        # Exit loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.5)  # Wait 0.5 seconds between captures

finally:
    # Release the cameras and destroy all OpenCV windows
    cams.end()
    cv2.destroyAllWindows()
