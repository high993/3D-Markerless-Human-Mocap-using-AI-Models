import cv2
import numpy as np

# Load the image from the specified path
image_path = r"C:\Users\Intel\Desktop\pythonproject5\backend\cam_calibration\setfloorcam_2\image_0.jpg"
image = cv2.imread(image_path)

# Check if the image was loaded successfully
if image is None:
    print(f"Error: Could not load image from path: {image_path}")
else:
    # Camera intrinsics
    camera_matrix = np.array([
        [516.0170622479519, 0.0, 296.6313429461038],
        [0.0, 516.433960550695, 226.67993714171007],
        [0.0, 0.0, 1.0]
    ])

    dist_coeffs = np.array([-0.09368630072910854, 0.35221838002636074, -0.004612092753702024, -0.003267042051470195, -0.7839345855344914])

    # Update the checkerboard pattern size to 6x4
    checkerboard_size = (6, 4)  # 6 by 4 checkerboard pattern

    # Find the checkerboard corners
    ret, corners = cv2.findChessboardCorners(image, checkerboard_size, None)

    if ret:
        # Define the real-world coordinates of the checkerboard points (Z=0 plane)
        square_size = 120  # Assuming each square is 30mm x 30mm
        objp = np.zeros((np.prod(checkerboard_size), 3), np.float32)
        objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
        objp *= square_size

        # Estimate the pose of the checkerboard
        _, rvec, tvec = cv2.solvePnP(objp, corners, camera_matrix, dist_coeffs)

        # Define the cube vertices using the full checkerboard area as the base
        cube_points_3D = np.array([
            [0, 0, 0],                        # Bottom face corners
            [checkerboard_size[0] * square_size, 0, 0],
            [checkerboard_size[0] * square_size, checkerboard_size[1] * square_size, 0],
            [0, checkerboard_size[1] * square_size, 0],
            [0, 0, -checkerboard_size[1] * square_size],   # Top face corners (height matches width)
            [checkerboard_size[0] * square_size, 0, -checkerboard_size[1] * square_size],
            [checkerboard_size[0] * square_size, checkerboard_size[1] * square_size, -checkerboard_size[1] * square_size],
            [0, checkerboard_size[1] * square_size, -checkerboard_size[1] * square_size]
        ], dtype=np.float32)

        # Project the 3D cube points onto the image
        cube_points_2D, _ = cv2.projectPoints(cube_points_3D, rvec, tvec, camera_matrix, dist_coeffs)

        # Draw the cube on the image
        cube_points_2D = np.int32(cube_points_2D).reshape(-1, 2)
        cv2.drawContours(image, [cube_points_2D[:4]], -1, (0, 255, 0), 3)  # Bottom face
        cv2.drawContours(image, [cube_points_2D[4:]], -1, (0, 255, 0), 3)  # Top face
        for i, j in zip(range(4), range(4, 8)):
            cv2.line(image, tuple(cube_points_2D[i]), tuple(cube_points_2D[j]), (255, 0, 0), 3)

        # Save or display the result
        output_image_path = r"C:\Users\Intel\Desktop\pythonproject5\backend\cam_calibration\setfloorcam_0\image_with_cube.jpg"
        cv2.imwrite(output_image_path, image)
        print(f"Cube drawn on the image and saved to: {output_image_path}")
    else:
        print("Checkerboard corners not found in the image.")
