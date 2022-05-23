import numpy as np
import cv2
import math
import dlib
from regex import I

class FaceNormalizer:
    def __init__(self):
        self.face_detector     = dlib.get_frontal_face_detector()
        self.landmark_detector = dlib.shape_predictor('resources/shape_predictor_68_face_landmarks.dat')

        self.output_size = (32, 32)

    def normalize(self, img: cv2.Mat, first: bool = True):
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
            
            rot_mat = cv2.getRotationMatrix2D(c_eye, alpha * 180.0 / math.pi + 180.0, 1.0)
            p33 = np.matmul(rot_mat[:, :2], np.array(landmarks[33]).reshape(-1, 1)).flatten() + np.array([rot_mat[0][2], rot_mat[1][2]])
            rot_mat[0][2] += w/2 - p33[0]
            rot_mat[1][2] += h/2 - p33[1]
            result = cv2.warpAffine(img, rot_mat, img.shape[1::-1])
            
            landmarks = np.array([np.matmul(rot_mat[:, :2], np.array([p]).reshape(-1, 1)).flatten() + np.array([rot_mat[0][2], rot_mat[1][2]]) for p in landmarks])
            rect = cv2.boundingRect(np.array(landmarks).astype(np.float32))
            result = result[max(rect[1] - int(rect[3] * 0.4), 0) : rect[1] + int(rect[3] * 0.9), rect[0] : rect[0] + rect[2]]
            result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            M = np.max(result)
            result = result / M

            res.append(cv2.resize(result, self.output_size))

        if first:
            if len(res):
                return res[0]
            return None
        return res
