3D Markerless Human-Mocap using AI Models 
This project contains the necessary computer code to transfer 2D motion from 4 cameras, 
triangulate it and then reproject it back into 3d space.

This code requires the pseyepy python library: https://github.com/bensondaled/pseyepy

It is recommended that the libusb be compiled separately for your operating environment and machine specifications. 

To activate the backend of the code run    python app.py   in a terminal located under the directory pythonproject5\backend
To launch the front end of the application, navigate to the     pythonproject5\pose    directory and run    yarn dev   
and copy and paste the URL into your browser. 

Begin by placing the 120mm 7 by 5 checker pattern down on the floor in the center of the room, gather the extrinsics and then compute the camera poses. 
If the 120mm doesn’t register because the scene is to large you can try getting a bigger checker 
pattern from here: https://markhedleyjones.com/projects/calibration-checkerboard-collection
Some modifications may need to be done to the location of the cameras under Mocap-Human\backend\camcalibratrion\entrinsic.py 
to get the exact poses.


Note: you will need to gather the intrinsic matrices and distortion coefficients for each camera 
separately and update the values in the script, here’s the link I used to get started  https://github.com/jyjblrd/Low-Cost-Mocap/discussions/11#discussioncomment-9380283
Some of the TESTSCRIPT pieces should help you diagnose and make sure that your setup is running 
correctly however some of them can be ignored entirely. 

TO DO:
-Make the directories local
-Update some of the referencing for the intrinsics 
-Optimize mediapipe pose estimation compatibility with Nvidia GPUs (cuda)
-Combine some of the features of the script so that the scene can be processed in real time or semi real time.
-implement Kalman filter for motion tracking smoothing.
- The code currently using uses mediapipe pose estimator however it could easily be swapped out for an alternative with more points for better accuracy.
- 
I realize that some of the functionalities of the code are inoperable at this time, I plan on fixing these issues going forward when I merge this project
with some others and when I upgrade to better hardware. For now, the code is setup to run on pretty much any machine because most of the compute is done in the post processing.
 I have a few good ideas as to what I’m going to flesh out this project into, but please feel free to leave a suggestion!



