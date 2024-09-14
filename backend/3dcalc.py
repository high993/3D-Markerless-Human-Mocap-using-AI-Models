import cv2
import numpy as np
import pandas as pd
import json
from scipy import linalg

def load_coordinates_from_csv(camera_id):
    """Load the (x, y) coordinates for all 33 points from the CSV file for a given camera."""
    file_path = f'keypoints/camera_{camera_id}_keypoints.csv'
    df = pd.read_csv(file_path)
    
    # Group by frame and aggregate points
    grouped = df.groupby('frame', group_keys=False).apply(lambda x: x[['point', 'x', 'y', 'visibility']].values.tolist())

    # Extract a list of points for each frame
    frames = []
    for frame_points in grouped:
        points = [(p[1], p[2], p[3]) if p[3] > 0.7 else (None, None, None) for p in frame_points]
        frames.append(points)
    
    return frames

def triangulate_point(image_points, camera_poses, camera_intrinsics):
    image_points = np.array(image_points)

    # Find indices where points are None
    none_indices = np.where(np.all(image_points == None, axis=1))[0]
    
    # Ensure the sizes are consistent before deletion
    if len(none_indices) > 0:
        image_points = np.delete(image_points, none_indices, axis=0)
        camera_poses = np.delete(camera_poses, none_indices, axis=0)
        camera_intrinsics = np.delete(camera_intrinsics, none_indices, axis=0)

    if len(image_points) <= 1:
        return [None, None, None]

    Ps = []  # Projection matrices

    for i, (camera_pose, intrinsic_matrix) in enumerate(zip(camera_poses, camera_intrinsics)):
        RT = np.c_[camera_pose["R"], camera_pose["t"]]
        P = intrinsic_matrix @ RT
        Ps.append(P)

    def DLT(Ps, image_points):
        A = []
        for P, image_point in zip(Ps, image_points):
            A.append(image_point[1] * P[2, :] - P[1, :])
            A.append(P[0, :] - image_point[0] * P[2, :])
        A = np.array(A).reshape((len(Ps) * 2, 4))
        B = A.transpose() @ A
        U, s, Vh = linalg.svd(B, full_matrices=False)
        object_point = Vh[3, 0:3] / Vh[3, 3]
        return object_point

    object_point = DLT(Ps, image_points)

    return object_point

def weighted_triangulate_point(image_points, camera_poses, camera_intrinsics, confidences):
    """Triangulate the point using weighted confidence scores for each camera."""
    valid_points = []
    valid_poses = []
    valid_intrinsics = []
    active_confidences = []
    
    # Filter out invalid points and corresponding poses, intrinsics, and confidences
    for i, point in enumerate(image_points):
        if point[0] is not None and point[1] is not None:
            valid_points.append(point[:2])  # (x, y) point
            valid_poses.append(camera_poses[i])
            valid_intrinsics.append(camera_intrinsics[i])
            active_confidences.append(confidences[i])

    if len(valid_points) < 2:
        return None  # Not enough points to triangulate

    # Normalize the confidence scores
    active_confidences = np.array(active_confidences)
    normalized_confidences = active_confidences / np.sum(active_confidences)

    # Compute the projection matrices
    Ps = []
    for pose, intrinsics in zip(valid_poses, valid_intrinsics):
        RT = np.c_[pose["R"], pose["t"]]
        P = intrinsics @ RT
        Ps.append(P)

    # Triangulate the points with confidence-weighted reprojection
    object_point = triangulate_point(valid_points, valid_poses, valid_intrinsics)

    # Ensure object_point is a numpy array if valid
    if object_point is not None:
        object_point = np.array(object_point)

    return object_point  # Return the triangulated 3D point

def save_3d_points(all_3d_points):
    """Save the formatted 3D points to a JSON file."""
    output_file = 'output_3d_points.json'
    with open(output_file, 'w') as f:
        json.dump(all_3d_points, f, indent=4)
    print(f"3D points saved to {output_file}.")

def main():
    # Load the points for each camera
    camera_0_points = load_coordinates_from_csv(0)
    camera_1_points = load_coordinates_from_csv(1)
    camera_2_points = load_coordinates_from_csv(2)
    camera_3_points = load_coordinates_from_csv(3)

    # Determine the minimum number of frames available across all cameras
    min_frames = min(len(camera_0_points), len(camera_1_points), len(camera_2_points), len(camera_3_points))
    print(f"Processing {min_frames} frames.")

    # Load intrinsic parameters from JSON file
    with open("calibration_data.json", "r") as f:
        intrinsic_data = json.load(f)

    # Load extrinsic parameters from JSON file
    with open(r"C:\Users\Intel\Desktop\pythonproject5\backend\camera_poses.json") as f:
        extrinsic_data = json.load(f)

    # Define camera intrinsics and extrinsics
    camera_poses = []
    camera_intrinsics = []
    for i in range(4):  # Assuming 4 cameras
        # Extract rvec and tvec from the JSON file and flatten them
        rvec = np.array(extrinsic_data[f"camera_{i}"]["rvec"]).flatten()
        tvec = np.array(extrinsic_data[f"camera_{i}"]["tvec"]).flatten()
        
        # Convert rvec to rotation matrix
        R, _ = cv2.Rodrigues(rvec)
        
        camera_poses.append({
            "R": R,  # Rotation matrix
            "t": tvec  # Translation vector
        })
        camera_intrinsics.append(np.array(intrinsic_data[f"camera_{i}"]["camera_matrix"]))

    # Prepare to collect 3D points
    all_3d_points = []

    # Triangulate the points for each frame and for all 33 points in that frame
    for frame_idx in range(min_frames):
        frame_3d_points = []
        for point_idx in range(33):  # 33 points per frame
            points_for_all_cameras = [
                camera_0_points[frame_idx][point_idx],
                camera_1_points[frame_idx][point_idx],
                camera_2_points[frame_idx][point_idx],
                camera_3_points[frame_idx][point_idx]
            ]

            # Extract visibility scores (confidence) for each point
            confidences = [point[2] if point[2] is not None else 0 for point in points_for_all_cameras]

            # Use the weighted triangulation function
            object_point = weighted_triangulate_point(points_for_all_cameras, camera_poses, camera_intrinsics, confidences)

            if object_point is not None:
                # Flip the Z-axis
                object_point[2] *= -1  # Multiply the Z value by -1 to move points above the plane
                frame_3d_points.append(object_point.tolist())  # Only call .tolist() on valid points
                print(f"Triangulated 3D point for frame {frame_idx}, point {point_idx}: {object_point}")
            else:
                frame_3d_points.append([None, None, None])  # No valid 3D point could be triangulated
                print(f"Skipping point {point_idx} in frame {frame_idx} due to insufficient data for triangulation.")
        
        all_3d_points.append(frame_3d_points)

    # Save the 3D points to a JSON file
    save_3d_points(all_3d_points)

if __name__ == "__main__":
    main()
