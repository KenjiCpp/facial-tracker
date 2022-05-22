import numpy as np
import cv2
import math
from utils import Mesh, Candide3, transform

candide = Candide3()

mesh = Mesh([[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.0, 0.5, 0.0]], [[0, 1, 2]])

ry = 0.0
rz = 0.0

while True:
    img = cv2.imread('resources/face_0.jpg')
    img, out = mesh.draw(img, transform.translate(0.0, 0.0, 3.0), transform.rotate(0.0, ry, rz))
    ry += 0.01
    rz += 0.002
    cv2.imshow('Render', cv2.resize(img, np.array([out.shape[1] // 2, out.shape[0] // 2])))
    cv2.imshow('Render2', cv2.resize(out, np.array([out.shape[1] // 2, out.shape[0] // 2])))
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()