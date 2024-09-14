import cv2
import numpy as np
import json

# Define the dimensions of the checkerboard
checkerboard_size = (6, 4)  # (columns, rows) of inner corners
square_size = 120  # Size of a square in the checkerboard (in millimeters)

# Define 3D object points for the checkerboard corners
objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# Camera calibration data for each camera (original distortion coefficients restored)
camera_data = {
    "camera_0": {
        "camera_matrix": np.array([
            [516.0170622479519, 0.0, 296.6313429461038],
            [0.0, 516.433960550695, 226.67993714171007],
            [0.0, 0.0, 1.0]
        ]),
        "distortion_coefficients": np.array([
            -0.09368630072910854, 0.35221838002636074, -0.004612092753702024, -0.003267042051470195, -0.7839345855344914
        ])
    },
    "camera_1": {
        "camera_matrix": np.array([
            [538.1317538879117, 0.0, 317.64096118891337],
            [0.0, 537.0117414868862, 223.53415500117256],
            [0.0, 0.0, 1.0]
        ]),
        "distortion_coefficients": np.array([
            -0.0526801543062888, -0.5868732370771725, 0.0010597147025117484, 0.0033764576298374854, 2.313185186898774
        ])
    },
    "camera_2": {
        "camera_matrix": np.array([
            [540.5349059925899, 0.0, 297.88131433792233],
            [0.0, 540.7905254879001, 231.73257538542006],
            [0.0, 0.0, 1.0]
        ]),
        "distortion_coefficients": np.array([
            -0.12114421742160625, 0.6055569602862708, -0.0037562909354312305, -0.0040356634736726585, -1.8625990867859497
        ])
    },
    "camera_3": {
        "camera_matrix": np.array([
            [537.6520952232842, 0.0, 277.7550675244408],
            [0.0, 536.4369700500476, 218.57866360661265],
            [0.0, 0.0, 1.0]
        ]),
        "distortion_coefficients": np.array([
            -0.08670335128609498, 0.026475786619421824, 0.0015977526998421844, -0.007318669419197426, 0.47146144753949526
        ])
    }
}

# List of image paths
image_paths = [
    r'C:\Users\Intel\Desktop\pythonproject5\backend\cam_calibration\setfloorcam_0\image_0.jpg',
    r'C:\Users\Intel\Desktop\pythonproject5\backend\cam_calibration\setfloorcam_1\image_0.jpg',
    r'C:\Users\Intel\Desktop\pythonproject5\backend\cam_calibration\setfloorcam_2\image_0.jpg',
    r'C:\Users\Intel\Desktop\pythonproject5\backend\cam_calibration\setfloorcam_3\image_0.jpg'
]

# Define the 3D points for the rectangular prism (5 by 3 squares)
prism_length_x = 5 * square_size  # 5 squares in x (600mm)
prism_length_y = 3 * square_size  # 3 squares in y (360mm)
prism_height = 3 * square_size    # Height of the prism (360mm)

prism_points = np.float32([
    [0, 0, 0],              # Point 0: Bottom-left corner of the base
    [0, prism_length_y, 0],  # Point 1: Top-left corner of the base
    [prism_length_x, prism_length_y, 0],  # Point 2: Top-right corner of the base
    [prism_length_x, 0, 0],  # Point 3: Bottom-right corner of the base
    [0, 0, -prism_height],   # Point 4: Bottom-left corner of the top
    [0, prism_length_y, -prism_height],  # Point 5: Top-left corner of the top
    [prism_length_x, prism_length_y, -prism_height], # Point 6: Top-right corner of the top
    [prism_length_x, 0, -prism_height]   # Point 7: Bottom-right corner of the top
])

# Define the 3D test point
test_point_3D = np.array([0, 0, 360], dtype=np.float32)  # For example, a point at (0, 0, 360)

# Dictionary to store rvecs and tvecs for each camera
camera_poses = {}

# Function to manually project a 3D point using the full projection matrix P = K[R | t]
def project_test_point_manual(camera_idx, rvecs, tvecs):
    camera_key = f"camera_{camera_idx}"
    camera_matrix = camera_data[camera_key]["camera_matrix"]
    
    # Convert rvec to rotation matrix
    R, _ = cv2.Rodrigues(rvecs)
    
    # Create the full projection matrix [R | t]
    Rt = np.hstack((R, tvecs))  # Combine R and t into a 3x4 matrix
    
    # Project the 3D point using the projection matrix
    point_3D_homogeneous = np.hstack([test_point_3D, [1]])  # Convert to homogeneous coordinates
    projected_point = camera_matrix @ Rt @ point_3D_homogeneous  # Apply projection matrix
    
    # Convert back from homogeneous coordinates to 2D
    projected_point_2D = projected_point[:2] / projected_point[2]
    return projected_point_2D.flatten()

# Loop through each image path
for idx, image_path in enumerate(image_paths):
    image = cv2.imread(image_path)

    if image is None:
        print(f"Failed to load image at {image_path}")
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

    if ret:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        # Apply flipping for cameras 2 and 3
        if idx == 2 or idx == 3:  # Cameras 2 and 3 are flipped
            corners_refined = corners_refined[::-1]  # Flip the order of corners (reverse the list may need to adjust in future setups see readme )
        
        # Choose the appropriate camera parameters
        camera_key = f"camera_{idx}"
        camera_matrix = camera_data[camera_key]["camera_matrix"]
        dist_coeffs = camera_data[camera_key]["distortion_coefficients"]

        # Find the rotation and translation vectors using solvePnP
        success, rvecs, tvecs = cv2.solvePnP(objp, corners_refined, camera_matrix, dist_coeffs)

        if not success:
            print(f"Could not solve PnP for image {idx}.")
            continue
        
        
        camera_poses[camera_key] = {
            "rvec": rvecs.tolist(),
            "tvec": tvecs.tolist()
        }

        # Project the prism points onto the image plane
        imgpts, _ = cv2.projectPoints(prism_points, rvecs, tvecs, camera_matrix, dist_coeffs)
        imgpts = np.int32(imgpts).reshape(-1, 2)

        # Print the 2D coordinates for each vertex
        print(f"2D Coordinates for Camera {idx}:")
        for i, point in enumerate(imgpts):
            print(f"Vertex {i}: {point}")
        
        # Manually project the test point using the full projection matrix
        test_point_2D = project_test_point_manual(idx, rvecs, tvecs)
        print(f"Test Point 2D coordinates for Camera {idx}: {test_point_2D}")
        print(f"Vertex 4 2D coordinates for Camera {idx} (from cube): {imgpts[4]}")

        # Compare with Vertex 4 coordinates
        print(f"Expected (same as Vertex 4) 2D coordinates for Camera {idx}: {imgpts[4]}")

        # Draw the prism
        image = cv2.drawContours(image, [imgpts[:4]], -1, (0, 0, 255), 3)  # Base of the prism
        for i in range(4):
            image = cv2.line(image, tuple(imgpts[i]), tuple(imgpts[i + 4]), (0, 255, 0), 3)  # Pillars
        image = cv2.drawContours(image, [imgpts[4:]], -1, (255, 0, 0), 3)  # Top of the prism

        # Label each vertex of the prism
        for i, point in enumerate(imgpts):
            cv2.putText(image, f"{i}", tuple(point), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        # Display the image with the labeled prism
        cv2.imshow(f'Labeled Prism on Checkerboard {idx}', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        print(f"Checkerboard pattern not found in image {idx}.")

# Save the camera poses (rvecs and tvecs) to a JSON file
with open('backend/camera_poses.json', 'w') as json_file:
    json.dump(camera_poses, json_file, indent=4)

print("Camera poses saved to camera_poses.json")
