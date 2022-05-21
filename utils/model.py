import cv2
import dlib
import numpy as np
from filterpy.kalman import ExtendedKalmanFilter

class Model:
    def __init__(self, n_feature_points):
        self.n_params  = n_feature_points + 7  # Number of parameters
        self.n_measure = 2 * n_feature_points  # Number of measurements

        self.params = np.random.random(self.n_params) * 5.0  # Model's parameters

        self.ekf = ExtendedKalmanFilter(self.n_params, self.n_measure)  # EKF update equation

    def get_params(self):
        return self.params

    def set_params(self, params):
        self.params = params

    def render(self, img):
        pass