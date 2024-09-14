import cv2
import numpy as np

# Define the dimensions of the checkerboard
checkerboard_size = (6, 4)  # (columns, rows) of inner corners

# Define the size of a square in your checkerboard (120mm in this example)
square_size = 120  # in millimeters

# Define 3D object points for the checkerboard corners
objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# List of image paths
image_paths = [
    r'C:\Users\Intel\Desktop\pythonproject5\backend\cam_calibration\setfloorcam_0\image_0.jpg',
    r'C:\Users\Intel\Desktop\pythonproject5\backend\cam_calibration\setfloorcam_1\image_1.jpg',
    r'C:\Users\Intel\Desktop\pythonproject5\backend\cam_calibration\setfloorcam_2\image_2.jpg',
    r'C:\Users\Intel\Desktop\pythonproject5\backend\cam_calibration\setfloorcam_3\image_3.jpg'
]

# Camera intrinsic parameters and distortion coefficients for Camera 0
camera_matrix_0 = np.array([
    [516.0170622479519, 0.0, 296.6313429461038],
    [0.0, 516.433960550695, 226.67993714171007],
    [0.0, 0.0, 1.0]
])

dist_coeffs_0 = np.array([
    -0.09368630072910854,
    0.35221838002636074,
    -0.004612092753702024,
    -0.003267042051470195,
    -0.7839345855344914
])

# Camera intrinsic parameters and distortion coefficients for Camera 1
camera_matrix_1 = np.array([
    [538.1317538879117, 0.0, 317.64096118891337],
    [0.0, 537.0117414868862, 223.53415500117256],
    [0.0, 0.0, 1.0]
])

dist_coeffs_1 = np.array([
    -0.09368630072910854,
    0.35221838002636074,
    -0.004612092753702024,
    -0.003267042051470195,
    -0.7839345855344914
])

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

# Define the 3D point to test: (0, 0, 0)
test_point_3D = np.array([[0, 0, 360]], dtype=np.float32)  # The point (0, 0, 0)

# Lists to store 2D points for Cameras 0 and 1
points_cam0 = []
points_cam1 = []

# Loop through each image path
for idx, image_path in enumerate(image_paths):
    image = cv2.imread(image_path)

    # Check if the image was loaded successfully
    if image is None:
        print(f"Failed to load image at {image_path}")
        continue

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

    # If found, refine the corner positions and proceed
    if ret:
        # Refining the corners detected to subpixel accuracy
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_refined = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria
        )
        
        # Draw and display the detected corners
        cv2.drawChessboardCorners(image, checkerboard_size, corners_refined, ret)
        cv2.imshow(f'Detected Corners {idx}', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Choose the correct camera matrix and distortion coefficients
        if idx == 0:
            camera_matrix = camera_matrix_0
            dist_coeffs = dist_coeffs_0
        elif idx == 1:
            camera_matrix = camera_matrix_1
            dist_coeffs = dist_coeffs_1
        else:
            camera_matrix = camera_matrix_0
            dist_coeffs = dist_coeffs_0
        
        # Find the rotation and translation vectors using solvePnP
        success, rvecs, tvecs = cv2.solvePnP(
            objp, corners_refined, camera_matrix, dist_coeffs
        )
        
        if not success:
            print(f"Could not solve PnP for image {idx}.")
            continue
        
        # Store the 2D points for Cameras 0 and 1
        if idx == 0:
            points_cam0 = corners_refined.astype(np.float32)
        elif idx == 1:
            points_cam1 = corners_refined.astype(np.float32)
            
        # Project the test point (0, 0, 0) onto the image plane for both cameras
        if idx == 0:
            # For Camera 0
            image_point_cam0, _ = cv2.projectPoints(test_point_3D, rvecs, tvecs, camera_matrix_0, dist_coeffs_0)
            print(f"Reprojected point (0, 0, 0) on Camera 0: {image_point_cam0.flatten()}")
        elif idx == 1:
            # For Camera 1
            image_point_cam1, _ = cv2.projectPoints(test_point_3D, rvecs, tvecs, camera_matrix_1, dist_coeffs_1)
            print(f"Reprojected point (0, 0, 0) on Camera 1: {image_point_cam1.flatten()}")

        # Project the prism points onto the image plane
        imgpts, _ = cv2.projectPoints(
            prism_points, rvecs, tvecs, camera_matrix, dist_coeffs
        )
        imgpts = np.int32(imgpts).reshape(-1, 2)

        # Draw the base of the prism
        image = cv2.drawContours(image, [imgpts[:4]], -1, (0, 0, 255), 3)

        # Draw the pillars of the prism
        for i in range(4):
            image = cv2.line(image, tuple(imgpts[i]), tuple(imgpts[i + 4]), (0, 255, 0), 3)

        # Draw the top of the prism
        image = cv2.drawContours(image, [imgpts[4:]], -1, (255, 0, 0), 3)

        # Label each vertex of the prism
        for i, point in enumerate(imgpts):
            cv2.putText(image, f"{i}", tuple(point), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Display the image with the labeled prism
        cv2.imshow(f'Labeled Prism on Checkerboard {idx}', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Optionally, save the image with the labeled prism
        output_path = f'output_image_with_labeled_prism_{idx}.jpg'
        cv2.imwrite(output_path, image)
    else:
        print(f"Checkerboard pattern not found in image {idx}.")

# Now we manually compute the essential matrix and recover R and t between Cameras 0 and 1
if points_cam0.size > 0 and points_cam1.size > 0:
    # Normalize the points using the camera intrinsics
    # Function to normalize 2D points using the camera matrix (intrinsics)
    def normalize_points(points, K):
        if len(points.shape) == 3:
            points = points.reshape(-1, 2)  # Flatten if points have an extra dimension
        # Add a third homogeneous coordinate (1s)
        points_hom = np.hstack([points, np.ones((points.shape[0], 1))])
        # Normalize by multiplying with the inverse of the camera matrix
        normalized_points = np.linalg.inv(K) @ points_hom.T
        return normalized_points[:2].T  # Return x and y in normalized coordinates


    points0_normalized = normalize_points(points_cam0, camera_matrix_0)
    points1_normalized = normalize_points(points_cam1, camera_matrix_1)

    # Construct the A matrix for the 8-point algorithm
    A = np.zeros((points_cam0.shape[0], 9))
    for i in range(points_cam0.shape[0]):
        x0, y0 = points0_normalized[i]
        x1, y1 = points1_normalized[i]
        A[i] = [x1 * x0, x1 * y0, x1, y1 * x0, y1 * y0, y1, x0, y0, 1]

    # Solve the linear system using SVD
    _, _, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3)

    # Enforce rank-2 constraint on F (make it rank 2 by zeroing the smallest singular value)
    U, S, Vt = np.linalg.svd(F)
    S[-1] = 0  # Set the smallest singular value to 0
    F = U @ np.diag(S) @ Vt

    # Compute the essential matrix
    E = camera_matrix_1.T @ F @ camera_matrix_0

    # Decompose the essential matrix to get R and t
    U, S, Vt = np.linalg.svd(E)
    W = np.array([[0, -1, 0],
                  [1,  0, 0],
                  [0,  0, 1]])

    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    t1 = U[:, 2]
    t2 = -U[:, 2]

    if np.linalg.det(R1) < 0:
        R1 = -R1
    if np.linalg.det(R2) < 0:
        R2 = -R2

    # Print the possible solutions
    print("Essential Matrix (E):\n", E)
    print("Possible Rotation Matrix (R1):\n", R1)
    print("Possible Rotation Matrix (R2):\n", R2)
    print("Possible Translation Vector (t1):\n", t1)
    print("Possible Translation Vector (t2):\n", t2)
else:
    print("Insufficient points to compute the essential matrix for Cameras 0 and 1.")


def triangulate_points(P1, P2, points1, points2):
    points4D_hom = cv2.triangulatePoints(P1, P2, points1.T, points2.T)
    
    # Check for invalid values in the homogeneous coordinate (points4D_hom[3])
    # Avoid division by values close to zero or infinite values
    epsilon = 1e-6  # Small value to avoid division by zero
    with np.errstate(invalid='ignore', divide='ignore'):
        points4D_hom[3] = np.where(np.abs(points4D_hom[3]) < epsilon, np.nan, points4D_hom[3])
        points4D = points4D_hom / points4D_hom[3]
    
    points3D = points4D[:3].T  # Return only the 3D coordinates (x, y, z)
    return points3D


# Projection matrices for the two cameras
P1 = np.hstack((np.eye(3), np.zeros((3, 1))))  # Camera 1 projection matrix
P2_R1_t1 = np.hstack((R1, t1.reshape(3, 1)))   # Camera 2 with R1 and t1
P2_R2_t2 = np.hstack((R2, t2.reshape(3, 1)))   # Camera 2 with R2 and t2

# Triangulate points for both solutions
points3D_R1_t1 = triangulate_points(P1, P2_R1_t1, points_cam0, points_cam1)
points3D_R2_t2 = triangulate_points(P1, P2_R2_t2, points_cam0, points_cam1)

# Check the z-coordinates of the triangulated points
if points3D_R1_t1 is not None and points3D_R1_t1.shape[1] == 3:
    if np.all(points3D_R1_t1[:, 2] > 0):
        print("R1 and t1 are the correct solution")
    else:
        print("R1 and t1 are incorrect")
else:
    print("Triangulation for R1 and t1 failed")

if points3D_R2_t2 is not None and points3D_R2_t2.shape[1] == 3:
    if np.all(points3D_R2_t2[:, 2] > 0):
        print("R2 and t2 are the correct solution")
    else:
        print("R2 and t2 are incorrect")
else:
    print("Triangulation for R2 and t2 failed")
