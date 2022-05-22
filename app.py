import cv2
import math
import numpy as np

from filterpy.kalman import ExtendedKalmanFilter
from utils.feature_point import FeaturePointsExtractor

cap = cv2.VideoCapture('resources/face.mp4')
if (cap.isOpened()== False):
    print("Error opening video stream or file")

frame_shape   = (int(cap.get(3)), int(cap.get(4)))
process_scale = 6

display_shape = (frame_shape[0] // 2, frame_shape[1] // 2)
process_shape = (frame_shape[0] // process_scale, frame_shape[1] // process_scale)

fpe = FeaturePointsExtractor()

is_first_frame   = True
n_feature_points = 48
patch_size       = 15
feature_points   = []

n_parameters   = n_feature_points + 7
n_measurements = 2 * n_feature_points
ekf = ExtendedKalmanFilter(n_parameters, n_measurements)

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        original = frame
        frame = cv2.resize(frame, process_shape)
        img   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if is_first_frame:
            feature_points = fpe.get_feature_points(img, n_feature_points)
            if feature_points is not None:
                is_first_frame = False

                # Handle first frame

                u  = feature_points.flatten().reshape(-1, 1)
                p  = np.random.random((n_parameters, 1)) # replace with a calculated initial parameters set
                pT = np.array([p.flatten()])

                upT = np.matmul(u, pT)
                ppT = np.matmul(p, pT)

                H = np.matmul(upT, np.linalg.inv(ppT + 0.01 * np.identity(ppT.shape[0])))

                ekf.update(feature_points.flatten().reshape(-1, 1), lambda state: H, lambda state: np.matmul(H, state))

        else:
            new_feature_points = fpe.get_correlated_feature_points(img_old, img, feature_points, patch_size)
            if new_feature_points is not None:

                # Handle next frame

                ekf.update(new_feature_points.flatten().reshape(-1, 1), lambda state: H, lambda state: np.matmul(H, state))

                feature_points = new_feature_points

        # Temporary, remove later

        # Calculate mean & var of feature points
        (mean_x, mean_y) = (0, 0)
        (var_x , var_y ) = (0, 0)
        for fp in feature_points:
            mean_x += fp[0]
            mean_y += fp[1]
        mean_x = mean_x / len(feature_points)
        mean_y = mean_y / len(feature_points)
        for fp in feature_points:
            var_x += (fp[0] - mean_x) ** 2
            var_y += (fp[1] - mean_y) ** 2
        var_x = math.sqrt(var_x / len(feature_points))
        var_y = math.sqrt(var_y / len(feature_points))

        alpha = 2.2
        min_x = mean_x - alpha * var_x
        min_y = mean_y - alpha * var_y
        max_x = mean_x + alpha * var_x
        max_y = mean_y + alpha * var_y

        original = cv2.rectangle(original, (process_scale * int(min_x), process_scale * int(min_y)), (process_scale * int(max_x), process_scale * int(max_y)), (255, 0, 0), 2)

        # Draw feature points
        for fp in feature_points:
            original = cv2.circle(original, (process_scale * fp[0], process_scale * fp[1]), 6, (0, 0, 255), -1)
        
        #print(ekf.x.flatten())

        # Display
        img_old = img
        cv2.imshow('App', cv2.resize(original, display_shape))
        
    else:
        is_first_frame = True
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()