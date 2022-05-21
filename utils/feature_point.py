import numpy as np
import cv2
import dlib

class FeaturePointsExtractor:
    def __init__(self):
        self.face_detector     = dlib.get_frontal_face_detector()
        self.landmark_detector = dlib.shape_predictor('resources/shape_predictor_68_face_landmarks.dat')

    def template_matching(self, img, tmp):
        match = cv2.matchTemplate(img, tmp, cv2.TM_CCOEFF_NORMED)
        minval, maxval, minpt, maxpt = cv2.minMaxLoc(match)
        return np.array(maxpt)

    def face_normalize(self, img):
        faces = self.face_detector(img, 1)
        if len(faces) == 0:
            return None, None

        face = faces[0]
        n_landmarks = 27
        p_landmarks = []
        landmarks = self.landmark_detector(img, face)
        for i in range(n_landmarks):
            p_landmarks.append((landmarks.part(i).x, landmarks.part(i).y))

        routes = []

        for i in range(15, -1, -1):
            from_coordinate = p_landmarks[i+1]
            to_coordinate = p_landmarks[i]
            routes.append(from_coordinate)

        from_coordinate = p_landmarks[0]
        to_coordinate = p_landmarks[17]
        routes.append(from_coordinate)

        for i in range(17, 20):
            from_coordinate = p_landmarks[i]
            to_coordinate = p_landmarks[i+1]    
            routes.append(from_coordinate)

        from_coordinate = p_landmarks[19]
        to_coordinate = p_landmarks[24]
        routes.append(from_coordinate)

        for i in range(24, 26):
            from_coordinate = p_landmarks[i]
            to_coordinate = p_landmarks[i+1]    
            routes.append(from_coordinate)

        from_coordinate = p_landmarks[26]
        to_coordinate = p_landmarks[16]
        routes.append(from_coordinate)
        routes.append(to_coordinate)

        mask = np.zeros((img.shape[0], img.shape[1]))
        mask = cv2.fillConvexPoly(mask, np.array(routes), 1)
        mask = mask.astype(np.bool)

        out = np.zeros_like(img)
        out[mask] = img[mask]

        return out, mask.astype(np.uint8)

    def get_feature_points(self, img, n):
        face, mask = self.face_normalize(img)
        print(type(mask))
        if face is None:
            return

        feature_points = cv2.goodFeaturesToTrack(img, n, 0.01, 0.0, mask=mask)
        feature_points = np.array(feature_points).astype(np.int32).reshape((n, 2))

        return feature_points

    def get_correlated_feature_points(self, img_old, img, feature_points, patch_size):
        res = []

        for fp in feature_points:
            patch = cv2.getRectSubPix(img_old, (patch_size, patch_size), fp.astype(np.float32))
            res.append(self.template_matching(img, patch) + np.array([patch_size // 2, patch_size // 2]))

        return np.array(res).astype(np.int32)
