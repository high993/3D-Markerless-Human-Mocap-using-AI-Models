import os
import cv2 as cv
import mediapipe as mp
import csv

# Paths to image directories for each camera (use absolute paths)
image_dirs = [
    r'C:\Users\Intel\Desktop\pythonproject5\backend\output_images\camera_0',
    r'C:\Users\Intel\Desktop\pythonproject5\backend\output_images\camera_1',
    r'C:\Users\Intel\Desktop\pythonproject5\backend\output_images\camera_2',
    r'C:\Users\Intel\Desktop\pythonproject5\backend\output_images\camera_3'
]

# MediaPipe Pose setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Output directory for keypoints CSVs
output_dir = 'keypoints'
os.makedirs(output_dir, exist_ok=True)

# Write headers to CSV files
headers = ['frame', 'point', 'x', 'y', 'visibility']

for camera_index, image_dir in enumerate(image_dirs):
    # Open CSV file for writing keypoints
    csv_path = os.path.join(output_dir, f'camera_{camera_index}_keypoints.csv')
    with open(csv_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(headers)

        # Process each image in the directory
        for frame_index, image_name in enumerate(sorted(os.listdir(image_dir))):
            image_path = os.path.join(image_dir, image_name)
            image = cv.imread(image_path)

            if image is None:
                print(f"Error: Could not read image {image_path}")
                continue

            height, width, _ = image.shape  # Get the dimensions of the image
            image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                for point_id, landmark in enumerate(results.pose_landmarks.landmark):
                    # Convert normalized coordinates to pixel values
                    x_pixel = int(landmark.x * width)
                    y_pixel = int(landmark.y * height)
                    csv_writer.writerow([frame_index, point_id, x_pixel, y_pixel, landmark.visibility])

        print(f"Finished processing images for camera {camera_index}.")

        # "Reset" the pose estimator to free memory before processing the next camera
        pose.close()
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose()

print("Processing complete for all cameras.")