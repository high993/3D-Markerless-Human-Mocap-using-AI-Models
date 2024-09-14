import numpy as np
import cv2 as cv

class KalmanFilter3D:
    def __init__(self, num_objects, dt=0.1, process_noise_cov=1e-2, measurement_noise_cov=1e-1):
        """
        Initializes a Kalman filter for tracking multiple 3D objects.

        Parameters:
        - num_objects: The number of objects to track.
        - dt: Time step between measurements.
        - process_noise_cov: Process noise covariance (default: 1e-2).
        - measurement_noise_cov: Measurement noise covariance (default: 1e-1).
        """
        self.num_objects = num_objects
        self.dt = dt

        # Define state dimensions: 3D position (x, y, z) and 3D velocity (vx, vy, vz)
        state_dim = 6
        measurement_dim = 3

        # Initialize Kalman filters for each object
        self.kalman_filters = []
        for _ in range(num_objects):
            kf = cv.KalmanFilter(state_dim, measurement_dim)
            kf.transitionMatrix = np.array([
                [1, 0, 0, dt, 0, 0],
                [0, 1, 0, 0, dt, 0],
                [0, 0, 1, 0, 0, dt],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1]
            ], dtype=np.float32)

            kf.measurementMatrix = np.eye(measurement_dim, state_dim, dtype=np.float32)

            kf.processNoiseCov = np.eye(state_dim, dtype=np.float32) * process_noise_cov
            kf.measurementNoiseCov = np.eye(measurement_dim, dtype=np.float32) * measurement_noise_cov
            kf.errorCovPost = np.eye(state_dim, dtype=np.float32)

            self.kalman_filters.append(kf)

    def predict(self):
        """
        Predicts the next state (3D position and velocity) for all objects.
        
        Returns:
        - predicted_states: List of predicted 3D positions for all objects.
        """
        predicted_states = []
        for kf in self.kalman_filters:
            predicted_state = kf.predict()
            predicted_states.append(predicted_state[:3].flatten())  # Extract the 3D position (x, y, z)

        return np.array(predicted_states)

    def correct(self, measurements):
        """
        Corrects the predicted state with the actual measurements.
        
        Parameters:
        - measurements: A list of 3D measurements (one per object).

        Returns:
        - corrected_states: List of corrected 3D positions for all objects.
        """
        corrected_states = []
        for i, kf in enumerate(self.kalman_filters):
            if measurements[i] is not None:
                measurement = np.array(measurements[i], dtype=np.float32).reshape(-1, 1)
                corrected_state = kf.correct(measurement)
                corrected_states.append(corrected_state[:3].flatten())  # Extract the 3D position (x, y, z))
            else:
                # If no measurement is available, use the prediction
                corrected_states.append(self.predict()[i])

        return np.array(corrected_states)

    def reset(self):
        """
        Resets all Kalman filters to their initial states.
        """
        for kf in self.kalman_filters:
            kf.statePost = np.zeros((6, 1), dtype=np.float32)
            kf.statePre = np.zeros((6, 1), dtype=np.float32)
            kf.errorCovPost = np.eye(6, dtype=np.float32)

# Example usage:
if __name__ == "__main__":
    num_objects = 33  # Example: 33 objects tracked by MediaPipe Pose Estimator
    kalman_filter = KalmanFilter3D(num_objects)

    # Simulated measurements (these would normally come from your 3D triangulation process)
    measurements = [np.random.rand(3) * 100 for _ in range(num_objects)]

    # Predict the next state
    predicted_positions = kalman_filter.predict()

    # Correct with actual measurements
    corrected_positions = kalman_filter.correct(measurements)

    print("Predicted Positions:\n", predicted_positions)
    print("Corrected Positions:\n", corrected_positions)
