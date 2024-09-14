import cv2
import numpy as np
import json

# Load the camera poses from the previously saved JSON file
with open('backend/camera_poses.json', 'r') as json_file:
    camera_poses = json.load(json_file)

# Camera intrinsic parameters
camera_data = {
    "camera_0": {
        "camera_matrix": np.array([
            [516.0170622479519, 0.0, 296.6313429461038],
            [0.0, 516.433960550695, 226.67993714171007],
            [0.0, 0.0, 1.0]
        ])
    },
    "camera_1": {
        "camera_matrix": np.array([
            [538.1317538879117, 0.0, 317.64096118891337],
            [0.0, 537.0117414868862, 223.53415500117256],
            [0.0, 0.0, 1.0]
        ])
    },
    "camera_2": {
        "camera_matrix": np.array([
            [540.5349059925899, 0.0, 297.88131433792233],
            [0.0, 540.7905254879001, 231.73257538542006],
            [0.0, 0.0, 1.0]
        ])
    },
    "camera_3": {
        "camera_matrix": np.array([
            [537.6520952232842, 0.0, 277.7550675244408],
            [0.0, 536.4369700500476, 218.57866360661265],
            [0.0, 0.0, 1.0]
        ])
    }
}

# Known 2D points for each camera (replace these with actual image coordinates or None if unavailable)
image_points_camera_0 = np.array([250.55536351, 371.36240081], dtype=np.float32)  # Example 2D point from Camera 0
image_points_camera_1 = np.array([189.85258153, 254.77832433], dtype=np.float32)  # Example 2D point from Camera 1
image_points_camera_2 = np.array([301.17091601, 363.60420965], dtype=np.float32)  # Example 2D point from Camera 2
image_points_camera_3 = np.array([385.1097811, 334.9713012], dtype=np.float32)  # Example 2D point from Camera 3


# Function to convert 2D points to homogeneous coordinates
def convert_to_homogeneous(points_2D):
    if points_2D is None:
        return None
    return np.array([[points_2D[0]], [points_2D[1]], [1]], dtype=np.float32)

# Convert 2D points to homogeneous coordinates
image_points_0 = convert_to_homogeneous(image_points_camera_0)
image_points_1 = convert_to_homogeneous(image_points_camera_1)
image_points_2 = convert_to_homogeneous(image_points_camera_2)
image_points_3 = convert_to_homogeneous(image_points_camera_3)

# Function to compute the projection matrix P = K[R|t]
def compute_projection_matrix(camera_matrix, rvec, tvec):
    R, _ = cv2.Rodrigues(np.array(rvec))  # Convert rvec to a rotation matrix
    Rt = np.hstack((R, np.array(tvec)))  # Combine rotation and translation
    P = camera_matrix @ Rt  # Compute the projection matrix
    return P

# Compute the projection matrices for each camera
P0 = compute_projection_matrix(camera_data["camera_0"]["camera_matrix"], camera_poses["camera_0"]["rvec"], camera_poses["camera_0"]["tvec"])
P1 = compute_projection_matrix(camera_data["camera_1"]["camera_matrix"], camera_poses["camera_1"]["rvec"], camera_poses["camera_1"]["tvec"])
P2 = compute_projection_matrix(camera_data["camera_2"]["camera_matrix"], camera_poses["camera_2"]["rvec"], camera_poses["camera_2"]["tvec"])
P3 = compute_projection_matrix(camera_data["camera_3"]["camera_matrix"], camera_poses["camera_3"]["rvec"], camera_poses["camera_3"]["tvec"])

# List to store available 2D points and corresponding projection matrices
points_2D = []
projection_matrices = []

# Check availability of each point and add to the list if available
if image_points_0 is not None:
    points_2D.append(image_points_0[:2].reshape(2, 1))
    projection_matrices.append(P0)

if image_points_1 is not None:
    points_2D.append(image_points_1[:2].reshape(2, 1))
    projection_matrices.append(P1)

if image_points_2 is not None:
    points_2D.append(image_points_2[:2].reshape(2, 1))
    projection_matrices.append(P2)

if image_points_3 is not None:
    points_2D.append(image_points_3[:2].reshape(2, 1))
    projection_matrices.append(P3)

# Ensure we have at least two valid points for triangulation
if len(points_2D) >= 2:
    # Triangulate the points to obtain the 3D coordinates
    points_4D = cv2.triangulatePoints(projection_matrices[0], projection_matrices[1], points_2D[0], points_2D[1])

    # Convert the 4D homogeneous points to 3D
    points_3D = points_4D[:3] / points_4D[3]

    # Reproject the 3D point back to each camera and compute reprojection error
    print("Computed 3D point:", points_3D.flatten())
    
    reprojection_errors = []
    for i, P in enumerate(projection_matrices):
        # Fix for hstack: Flatten points_3D before concatenating the homogeneous coordinate
        projected_2D = P @ np.hstack([points_3D.flatten(), 1])
        projected_2D = projected_2D[:2] / projected_2D[2]  # Convert back to 2D

        # Compute the reprojection error for this camera
        error = np.linalg.norm(points_2D[i] - projected_2D.reshape(2, 1))
        reprojection_errors.append(error)
        print(f"Reprojection error for Camera {i}: {error:.4f}")

    # Compute the average reprojection error
    avg_reprojection_error = np.mean(reprojection_errors)
    print(f"Average Reprojection Error: {avg_reprojection_error:.4f}")
    
else:
    print("Not enough valid points for triangulation.")
