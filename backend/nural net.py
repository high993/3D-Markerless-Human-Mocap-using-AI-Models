import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Step 1: Load the 3D points data from the JSON file
file_path = r'C:\Users\Intel\Desktop\pythonproject5\backend\output_3d_points.json'

with open(file_path, 'r') as file:
    data = json.load(file)

data_array = np.array(data)

# Step 2: Data validation - check for None values
def validate_data(data_array):
    """Check if any points are None and replace them with [0, 0, 0]."""
    for frame_idx in range(data_array.shape[0]):
        for point_idx in range(data_array.shape[1]):
            point = data_array[frame_idx, point_idx]
            if point is None or any([coord is None for coord in point]):
                print(f"Invalid point found at frame {frame_idx}, point {point_idx}: {point}")
                data_array[frame_idx, point_idx] = [0.0, 0.0, 0.0]  # Replace with [0, 0, 0]

# Validate the data
validate_data(data_array)

# Step 3: Rotation and scaling functions

def rotate_points(points, theta_x=0, theta_y=0, theta_z=0):
    """Rotates the point cloud by the given angles (theta_x, theta_y, theta_z) in radians."""
    
    # Rotation matrices for x, y, and z axes
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(theta_x), -np.sin(theta_x)],
                    [0, np.sin(theta_x), np.cos(theta_x)]])
    
    R_y = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                    [0, 1, 0],
                    [-np.sin(theta_y), 0, np.cos(theta_y)]])
    
    R_z = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                    [np.sin(theta_z), np.cos(theta_z), 0],
                    [0, 0, 1]])
    
    # Apply the rotations sequentially
    rotated_points = np.dot(points, R_x)
    rotated_points = np.dot(rotated_points, R_y)
    rotated_points = np.dot(rotated_points, R_z)
    
    return rotated_points

def scale_points(points, scale_factor):
    """Scales the point cloud by the given scale factor."""
    return points * scale_factor

# Step 4: Apply transformations to the point cloud

# Define rotation angles (in radians) and scaling factor
theta_x = np.radians(30)  # Rotate 30 degrees around the x-axis
theta_y = np.radians(45)  # Rotate 45 degrees around the y-axis
theta_z = np.radians(60)  # Rotate 60 degrees around the z-axis
scale_factor = 1.5        # Scale the points by 1.5

# Apply the transformations to each frame of data
num_frames = data_array.shape[0]  # Number of frames in the sequence

for frame_idx in range(num_frames):
    # Rotate points in each frame
    data_array[frame_idx] = rotate_points(data_array[frame_idx], theta_x, theta_y, theta_z)
    
    # Scale points (optional, but you mentioned the relative structure must remain intact)
    data_array[frame_idx] = scale_points(data_array[frame_idx], scale_factor)

# Step 5: Reshape the data for LSTM input

# Each frame has 33 points with 3 coordinates, so we flatten to (frames, 33 * 3) = (frames, 99)
data_array_reshaped = data_array.reshape(num_frames, -1)

# Define the number of timesteps (sequence length)
timesteps = 300  # Assuming sequences are 300 frames long

# Create sequences of 300 frames for LSTM input
X_train = np.array([data_array_reshaped[i:i+timesteps] for i in range(0, num_frames - timesteps)])
y_train = X_train  # For reconstruction, output is the same as input

# Step 6: Define the LSTM model
model = Sequential()

# LSTM layer (input_shape = (timesteps, features))
model.add(LSTM(units=128, return_sequences=True, input_shape=(timesteps, 99)))

# Additional LSTM layers (optional)
model.add(LSTM(units=64, return_sequences=False))

# Fully connected output layer for 99 features (33 points * 3 coords)
model.add(Dense(99))

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Print the model summary
model.summary()

# Step 7: Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Step 8: Make predictions on the training data
predictions = model.predict(X_train)

# Reshape the predictions back to (frames, points, 3) format for easier interpretation
predictions_reshaped = predictions.reshape(-1, 33, 3)

# Step 9: Save the predictions to a new JSON file
output_file_path = r'C:\Users\Intel\Desktop\pythonproject5\backend\generated_output_3d_points.json'

# Convert the NumPy array back to a list for JSON serialization
predictions_list = predictions_reshaped.tolist()

# Save to a new JSON file
with open(output_file_path, 'w') as output_file:
    json.dump(predictions_list, output_file)

print(f"Predicted 3D points saved to {output_file_path}")


