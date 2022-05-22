import numpy as np
import cv2
import math
import dlib
from regex import I

class FaceNormalizer:
    def __init__(self):
        self.face_detector     = dlib.get_frontal_face_detector()
        self.landmark_detector = dlib.shape_predictor('resources/shape_predictor_68_face_landmarks.dat')

        self.output_size = (64, 64)

    def normalize(self, img: cv2.Mat):
        h = img.shape[0]
        w = img.shape[1]
        faces = self.face_detector(img, 1)

        res  = []
        for face in faces:
            lm = self.landmark_detector(img, face)
            landmarks = np.array([[lm.part(i).x, lm.part(i).y] for i in range(68)])

            l_eye = np.mean(landmarks[36 : 36 + 6], axis=0)
            r_eye = np.mean(landmarks[42 : 42 + 6], axis=0)
            c_eye = 0.5 * (l_eye + r_eye)
            d_eye = l_eye - r_eye
            alpha = math.atan2(d_eye[1], d_eye[0])
            
            rot_mat = cv2.getRotationMatrix2D(c_eye, alpha * 180.0 / math.pi + 180, 1.0)
            p33 = np.matmul(rot_mat[:, :2], np.array(landmarks[33]).reshape(-1, 1)).flatten() + np.array([rot_mat[0][2], rot_mat[1][2]])
            rot_mat[0][2] += w/2 - p33[0]
            rot_mat[1][2] += h/2 - p33[1]
            result = cv2.warpAffine(img, rot_mat, img.shape[1::-1])
            
            landmarks = np.array([np.matmul(rot_mat[:, :2], np.array([p]).reshape(-1, 1)).flatten() + np.array([rot_mat[0][2], rot_mat[1][2]]) for p in landmarks])
            min_x = w
            max_x = -1
            min_y = h
            max_y = -1
            for p in landmarks:
                min_x = min(min_x, int(p[0]))
                max_x = max(max_x, int(p[0]))
                min_y = min(min_y, int(p[1]))
                max_y = max(max_y, int(p[1]))

            routes = []
            outer_landmarks = landmarks[:27]
            for i in range(15, -1, -1):
               from_coordinate = outer_landmarks[i+1]
               to_coordinate = outer_landmarks[i]
               routes.append(from_coordinate)
            from_coordinate = outer_landmarks[0]
            to_coordinate = outer_landmarks[17]
            routes.append(from_coordinate)
            for i in range(17, 20):
               from_coordinate = outer_landmarks[i]
               to_coordinate = outer_landmarks[i+1]
               routes.append(from_coordinate)
            from_coordinate = outer_landmarks[19]
            to_coordinate = outer_landmarks[24]
            routes.append(from_coordinate)
            for i in range(24, 26):
               from_coordinate = outer_landmarks[i]
               to_coordinate = outer_landmarks[i+1]
               routes.append(from_coordinate)
            from_coordinate = outer_landmarks[26]
            to_coordinate = outer_landmarks[16]
            routes.append(from_coordinate)
            routes.append(to_coordinate)

            mask = np.zeros(result.shape)
            mask = cv2.fillConvexPoly(mask, np.array(routes).astype(np.int32), (255, 255, 255))
            mask = mask.astype(np.bool)
 
            out = np.zeros_like(img)
            out[mask] = result[mask]
            out = out[min_y : max_y, min_x : max_x]
            out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)

            res.append(cv2.resize(out, self.output_size))

        return res
