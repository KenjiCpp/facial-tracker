import cv2
import numpy as np

from filterpy.kalman import ExtendedKalmanFilter
from utils.feature_point import FeaturePointsExtractor

cap = cv2.VideoCapture('resources/face.mp4')
if (cap.isOpened()== False):
    print("Error opening video stream or file")

frame_shape   = (int(cap.get(3)), int(cap.get(4)))
display_shape = (frame_shape[0] // 2, frame_shape[1] // 2)
process_shape = (frame_shape[0] // 8, frame_shape[1] // 8)

fpe = FeaturePointsExtractor()

is_first_frame   = True
n_feature_points = 10
patch_size       = 11
feature_points   = []

n_parameters   = n_feature_points + 7
n_measurements = n_feature_points * 2
ekf = ExtendedKalmanFilter(n_parameters, n_measurements)

face_cascade = cv2.CascadeClassifier('resources/haarcascade_frontalface_default.xml')
x = None
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        frame = cv2.resize(frame, process_shape)
        img   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if is_first_frame:
            is_first_frame = False
            feature_points = fpe.get_feature_points(img, n_feature_points)
        else:
            new_feature_points = fpe.get_correlated_feature_points(img_old, img, feature_points, patch_size)
            feature_points = new_feature_points

        for fp in feature_points:
            frame = cv2.circle(frame, fp, 1, (0, 0, 255), -1)
        
        img_old = img
        cv2.imshow('App', cv2.resize(frame, display_shape))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()