import cv2
import dlib
import numpy as np

from filterpy.kalman import ExtendedKalmanFilter

class FaceModel:
    def __init__(self, n_features: int = 28, patch_size: int = 25):
        # Face detector and landmark detector
        self.face_detector     = dlib.get_frontal_face_detector()
        self.landmark_detector = dlib.shape_predictor('resources/shape_predictor_68_face_landmarks.dat')

        # Parameters for feature extracting and tracking
        self.patch_size = patch_size + bool(patch_size % 2 == 0)
        self.n_features = n_features

        # Number of parameters and measurements
        self.n_params   = self.n_features + 7
        self.n_measures = 2 * self.n_features
        self.ekf        = ExtendedKalmanFilter(self.n_params, self.n_measures)  

        # Model parameters: [z, t, r, f]
        self.params = np.random.random(self.n_params).reshape(-1, 1)
        assert(len(self.params) == self.n_params)

        # Keep track of the measurements
        self.measure        = None
        self.measure_sum    = np.zeros(self.n_measures)
        self.measure_sumsqr = np.zeros(self.n_measures)
        self.measure_count  = 0


    def template_matching(self, img: cv2.Mat, tmp: cv2.Mat):
        (h, w) = tmp.shape
        match = cv2.matchTemplate(img, tmp, cv2.TM_CCOEFF_NORMED)
        minval, maxval, minpt, maxpt = cv2.minMaxLoc(match)
        return np.array([maxpt[0] + w // 2, maxpt[1] + w // 2])

    def extract_feature_points(self, img: cv2.Mat):
        faces = self.face_detector(img, 1)
        if len(faces) == 0:
            return None
        face = faces[0]
        mask = np.zeros(img.shape[:2],np.uint8)
        mask[face.top() : face.bottom(), face.left() : face.right()] = 255
        res = cv2.goodFeaturesToTrack(img, self.n_features, 0.01, 0)
        res = np.array(res).reshape((-1, 2))
        return np.array(res).astype(np.float32)

    def track_feature_points(self, img_src: cv2.Mat, img_dst: cv2.Mat, points: np.ndarray):
        res = []
        for fp in points:
            patch = cv2.getRectSubPix(img_src, (self.patch_size, self.patch_size), fp.astype(np.float32))
            res.append(self.template_matching(img_dst, patch))
        return np.array(res)
        
    def render(self, img: cv2.Mat):
        t = np.array([
            [1.0, 0.0, self.params[self.n_features + 0][0] - 20.0],
            [0.0, 1.0, self.params[self.n_features + 1][0] + 10.0]
        ])
        shifted = cv2.warpAffine(img, t, (img.shape[1], img.shape[0]))
        return shifted

    def update(self, measure: np.ndarray):
        self.measure_sum    = self.measure_sum + measure.flatten()
        self.measure_sumsqr = self.measure_sumsqr + measure.flatten() * measure.flatten()
        self.measure_count += 1

        if self.measure_count > 3:
            mean    = self.measure_sum / self.measure_count
            meansqr = self.measure_sumsqr / self.measure_count
            sqrvar  = meansqr - mean * mean

            R = np.diag(sqrvar)
            R = np.nan_to_num(R) + np.identity(self.n_measures) * 1e-100

            def HJacobian(params):
                res = np.eye(self.n_measures, self.n_params) * 1e-3
                return res

            def Hx(params):
                return self.measure
        
            self.ekf.predict_update(measure, HJacobian, Hx)
            self.params = self.ekf.x

        self.measure = measure
