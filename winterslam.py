import cv2
import numpy as np
from collections import deque

video_path = r"C:\Users\harsh\Downloads\video.mp4"
video = cv2.VideoCapture(video_path)

h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))

scale = 640 / w
new_w = 640
new_h = int(h * scale)

focal_val = 3840 * scale
cx = new_w / 2
cy = new_h / 2

K = np.array([[focal_val, 0, cx],
              [0, focal_val, cy],
              [0, 0, 1]], dtype=np.float64)

traj_canvas = np.zeros((1200, 1200, 3), dtype=np.uint8)
orb = cv2.ORB_create(nfeatures=3000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

cv2.namedWindow("ORB Matches", cv2.WINDOW_NORMAL)
cv2.namedWindow("Trajectory", cv2.WINDOW_NORMAL)

offset_x = 600
offset_z = int(1200 * 0.85)
prev_draw = (offset_x, offset_z)
cv2.circle(traj_canvas, prev_draw, 6, (255, 200, 0), -1)

heading = 0.0
pos_x   = 0.0
pos_z   = 0.0

hist = deque(maxlen=3)

prev_frame = None
frame_count = 0
FRAME_SKIP = 5

while True:
    ret, frame = video.read()
    if not ret:
        break

    frame_count += 1
    frame = cv2.resize(frame, (new_w, new_h))
    cv2.imshow("ORB Matches", frame)

    if frame_count % FRAME_SKIP != 0:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    if prev_frame is None:
        prev_frame = frame.copy()
        continue

    gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    if des1 is None or des2 is None or len(des1) < 8 or len(des2) < 8:
        prev_frame = frame.copy()
        continue

    matches = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]

    if len(good) < 8:
        prev_frame = frame.copy()
        continue

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

    E, mask = cv2.findEssentialMat(pts1, pts2, K,
                                   method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None or E.shape != (3, 3):
        prev_frame = frame.copy()
        continue

    mask = mask.ravel().astype(bool)
    if mask.sum() < 15 or mask.sum() / len(mask) < 0.6:
        prev_frame = frame.copy()
        continue

    _, R, t, _ = cv2.recoverPose(E, pts1[mask], pts2[mask], K)

    v1 = np.arctan2(R[0, 2], R[2, 2])     

    movement = v1

    if abs(np.degrees(movement)) > 10:
        prev_frame = frame.copy()
        continue

    hist.append(movement)

    if len(hist) == 3:
        signs = [np.sign(y) for y in hist]
        magnitudes = [abs(y) for y in hist]
        all_same_direction = len(set(signs)) == 1
        avg_magnitude = np.mean(magnitudes)

        if all_same_direction and avg_magnitude > np.radians(0.8):
            clean = float(np.mean(hist))
        else:
            clean = float(np.median(hist))
    else:
        clean = float(np.median(hist))


    heading += clean

    pos_x += np.sin(heading)
    pos_z += np.cos(heading)

    draw_x = int( pos_x * 0.08) + offset_x
    draw_z = int(-pos_z * 0.08) + offset_z

    draw_x = np.clip(draw_x, 0, 1199)
    draw_z = np.clip(draw_z, 0, 1199)

    curr_draw = (draw_x, draw_z)
    cv2.line(traj_canvas, prev_draw, curr_draw, (0, 255, 120), 2)
    cv2.circle(traj_canvas, (offset_x, offset_z), 6, (255, 200, 0), -1)
    prev_draw = curr_draw

    cv2.imshow("Trajectory", traj_canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    prev_frame = frame.copy()

video.release()
cv2.destroyAllWindows()
