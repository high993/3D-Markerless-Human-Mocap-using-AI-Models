import cv2
from pseyepy import Camera, Display

# Initialize two cameras with specific parameters
cameras = Camera([0, 1, 2, 3], fps=60, resolution=Camera.RES_LARGE, colour=True)

# Create a window for displaying the camera feeds
cv2.namedWindow("Camera 0", cv2.WINDOW_NORMAL)
cv2.namedWindow("Camera 1", cv2.WINDOW_NORMAL)
cv2.namedWindow("Camera 2", cv2.WINDOW_NORMAL)
cv2.namedWindow("Camera 3", cv2.WINDOW_NORMAL)
try:
    while True:
        # Read frames from both cameras
        frames, timestamps = cameras.read()

        # Display the frames
        cv2.imshow("Camera 0", frames[0])
        cv2.imshow("Camera 1", frames[1])
        cv2.imshow("Camera 2", frames[2])
        cv2.imshow("Camera 3", frames[3])
        
        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # When finished, close the camera and windowsq
    cameras.end()
    cv2.destroyAllWindows()
