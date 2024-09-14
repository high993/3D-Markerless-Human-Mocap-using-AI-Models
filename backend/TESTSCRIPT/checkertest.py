import cv2
import numpy as np

# Define the correct dimensions of the checkerboard
CHECKERBOARD = (6, 4)  # 6 inner corners horizontally and 4 inner corners vertically

# Load the image from the specified path
image_path = r'C:\Users\Intel\Desktop\pythonproject5\backend\cam_calibration\setfloorcam_1\image_11.jpg'
img = cv2.imread(image_path)

if img is None:
    print(f"Failed to load image from {image_path}")
else:
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Display the grayscale image for manual inspection
    cv2.imshow('Grayscale Image', gray)
    cv2.waitKey(0)

    # Attempt to find the chessboard corners on the grayscale image
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, 
                                             cv2.CALIB_CB_NORMALIZE_IMAGE)

    if ret:
        # Refine the corner positions to subpixel accuracy
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # Draw and display the detected corners
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow('Detected Chessboard', img)
        cv2.waitKey(0)  # Wait for a key press to close the window
    else:
        print("Chessboard not detected")

    cv2.destroyAllWindows()


r'C:\Users\Intel\Desktop\pythonproject5\backend\cam_calibration\setfloorcam_0\image_0.jpg'