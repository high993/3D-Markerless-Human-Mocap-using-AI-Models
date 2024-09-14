import cv2
import matplotlib.pyplot as plt

# File paths for the images from 4 cameras
image_paths = [
    r'C:\Users\Intel\Desktop\pythonproject5\backend\cam_calibration\setfloorcam_0\image_0.jpg',
    r'C:\Users\Intel\Desktop\pythonproject5\backend\cam_calibration\setfloorcam_1\image_0.jpg',
    r'C:\Users\Intel\Desktop\pythonproject5\backend\cam_calibration\setfloorcam_2\image_0.jpg',
    r'C:\Users\Intel\Desktop\pythonproject5\backend\cam_calibration\setfloorcam_3\image_0.jpg'
]

# Reprojected points for the 4 cameras
points = [
    (398,177),  # Point for Camera 0
    (255,257),   # Point for Camera 1
    (289,206),   # Example point for Camera 2
    (205,225)    # Example point for Camera 3
]

# Load and process each image
for i, image_path in enumerate(image_paths):
    # Load image
    image = cv2.imread(image_path)
    
    # Check if image loaded correctly
    if image is None:
        print(f"Failed to load Camera {i} image from {image_path}")
        continue
    
    # Draw the reprojected point on the image
    point = points[i]
    cv2.circle(image, (int(point[0]), int(point[1])), 10, (0, 0, 255), -1)  # Red dot
    
    # Display the image with the point
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(f'Camera {i} with reprojected point')
    plt.show()
