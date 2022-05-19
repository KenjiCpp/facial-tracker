import numpy as np
import cv2
from multiprocessing import Pool

class FeaturePointsExtractor:
    def __init__(self):
        self.__face_cascade = cv2.CascadeClassifier('resources/haarcascade_frontalface_default.xml')

    def get_feature_points(self, img, n):
        (x, y, w, h) = self.__face_cascade.detectMultiScale(img, 1.1, 4)[0]
        face = img[y : y + h, x : x + w]

        feature_points = cv2.goodFeaturesToTrack(img, n, 0.05, 10)
        feature_points = (np.array(feature_points) + np.array([x, y])).reshape((n, 2)).astype(np.int32)

        return feature_points

    def get_correlated_feature_points(self, img_old, img, feature_points, patch_size):
        (x, y, w, h) = self.__face_cascade.detectMultiScale(img_old, 1.1, 4)[0]
        face = img_old[y : y + h, x : x + w]

        res = []

        for fp in feature_points:
            patch = cv2.getRectSubPix(img, (patch_size, patch_size), fp.astype(np.float32))
            match = cv2.matchTemplate(face, patch, cv2.TM_CCOEFF_NORMED)
            min_value, max_value, pt_min, pt_max = cv2.minMaxLoc(match)
            res.append(np.array(pt_max) + np.array([x, y]) + (0.5 * np.array([patch_size, patch_size])))

        return np.array(res).astype(np.int32)
            
        