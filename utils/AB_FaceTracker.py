import numpy as np
import cv2
import math
from utils.Candide3 import Candide3

class AB_FaceTracker:
    def __init__(self):
        self.candide3 = Candide3()

        self.n_params = self.candide3.n_AUs + 3 + 3
        self.params   = np.zeros(self.n_params)

        self.n_static_params = self.candide3.n_SUs
        self.static_params   = np.zeros(self.n_static_params)

    def get_params(self):
        alpha     = self.params[0 : self.candide3.n_AUs]
        translate = self.params[self.candide3.n_AUs : self.candide3.n_AUs + 3]
        rotate    = self.params[self.candide3.n_AUs + 3 : ]
        return alpha, translate, rotate

