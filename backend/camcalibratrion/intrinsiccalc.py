import cv2
import numpy as np
import os
import glob
import json
import sys  # For getting the camera index from arguments

# Get camera index (i) from command line arguments
if len(sys.argv) > 1:
    i = int(sys.argv[1])  # Camera index passed as an argument
else:
    i = 2  # Default camera index

# Set directories for raw images and calibrated images
cam_images_folder_name = os.path.join(r'C:\Users\Intel\Desktop\pythonproject5\cam_calibration_intrinsic', f'intrinsiccam_{i}')
cam_images_folder_name_calibrated = f'{cam_images_folder_name}_c'
os.makedirs(cam_images_folder_name_calibrated, exist_ok=True)

CHECKERBOARD = (5, 6)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objpoints = []
imgpoints = []

# Define the real-world coordinates for the checkerboard points (in centimeters)
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

prev_img_shape = None

images = glob.glob(f'{cam_images_folder_name}/*.jpg')
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(
        gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
     
    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
     
    new_frame_name = os.path.join(cam_images_folder_name_calibrated, os.path.basename(fname))
    cv2.imwrite(new_frame_name, img)

if len(imgpoints) > 0:
    h, w = img.shape[:2]
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
 
    print("Camera matrix : \n", mtx)
    print("dist : \n", dist)
    
    calibration_data = {
        f"camera_{i}": {
            "camera_matrix": mtx.tolist(),
            "distortion_coefficients": dist.tolist()
        }
    }

    # Load existing data from JSON file if it exists
    json_file = os.path.join(r'C:\Users\Intel\Desktop\pythonproject5\backend', "calibration_data.json")
    if os.path.exists(json_file):
        with open(json_file, "r") as f:
            data = json.load(f)
    else:
        data = {}

    # Update the data with the new calibration data
    data.update(calibration_data)

    # Save the updated data back to the JSON file
    with open(json_file, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Calibration data saved to {json_file}")
else:
    print("No chessboard corners found, calibration not performed.")
