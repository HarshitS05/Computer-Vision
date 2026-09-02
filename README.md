Winter SLAM — Visual Odometry
A monocular visual odometry system that estimates a car's position and heading from dashcam footage using ORB feature matching and Essential Matrix decomposition.

Problem Statement
The car's LiDAR sensor is unavailable. Using only a front-facing dashcam, the system tracks the car's positional changes (x, z) and orientation (heading angle) relative to its starting point (0, 0) with an initial facing direction of 0°.

Approach

1. Feature Detection — ORB keypoints are extracted from each sampled frame
2. Pose Estimation — Essential Matrix computed via RANSAC, decomposed into rotation R and translation t using recoverPose
3. Heading Update — Rotation angle extracted from R and accumulated over frames
4. Trajectory Visualization — Path drawn incrementally on a canvas using OpenCV

Outlier Filtering

To reduce drift and noise:
- Frames with fewer than 8 good matches are skipped
- Frames where inlier ratio < 60% are skipped
- Heading changes > 10° per step are discarded
- A 3-frame rolling window applies median smoothing; if all 3 frames agree in direction and magnitude > 0.8°, the mean is used instead

Requirements

Python 3.8+
opencv-python
numpy

Install dependencies:
bash
pip install opencv-python numpy


1. Set the path to your dashcam video:
-python
-video_path = r"path/to/your/video.mp4"


2. Run the script:
bash
python winter_slam.py


Expected Output

- Dashcam — live dashcam feed
- Trajectory — estimated path of the car (yellow dot = start)


File Structure

winter_slam.py    
README.md         



