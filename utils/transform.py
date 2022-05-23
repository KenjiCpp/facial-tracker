import numpy as np
import math

def normalize(v: np.ndarray):
    norm=np.linalg.norm(v)
    if norm==0:
        norm=np.finfo(v.dtype).eps
    return v/norm

def translate(tx: float, ty: float, tz: float):
    return np.array([[ 1.0,  0.0,  0.0,  tx  ],
                     [ 0.0,  1.0,  0.0,  ty  ],
                     [ 0.0,  0.0,  1.0,  tz  ],
                     [ 0.0,  0.0,  0.0,  1.0 ]])

def rotate(rx: float, ry: float, rz: float):
    sin_rx = math.sin(rx)
    sin_ry = math.sin(ry)
    sin_rz = math.sin(rz)

    cos_rx = math.cos(rx)
    cos_ry = math.cos(ry)
    cos_rz = math.cos(rz)

    Rx = np.array([[ 1.0,  0.0   ,  0.0   ,  0.0 ], 
                   [ 0.0,  cos_rx, -sin_rx,  0.0 ], 
                   [ 0.0,  sin_rx,  cos_rx,  0.0 ],
                   [ 0.0,  0.0   ,  0.0   ,  1.0 ]])

    Ry = np.array([[ cos_ry,  0.0,  sin_ry,  0.0 ], 
                   [ 0.0   ,  1.0,  0.0   ,  0.0 ],
                   [-sin_ry,  0.0,  cos_ry,  0.0 ],
                   [ 0.0   ,  0.0,  0.0   ,  1.0 ]])

    Rz = np.array([[ cos_rz, -sin_rz,  0.0,  0.0 ], 
                   [ sin_rz,  cos_rz,  0.0,  0.0 ], 
                   [ 0.0   ,  0.0   ,  1.0,  0.0 ],
                   [ 0.0   ,  0.0   ,  0.0,  1.0 ]])

    return np.matmul(Rz, np.matmul(Ry, Rx))

def perspective(fov: float, aspect: float, near: float, far: float):
    b = 1.0 / math.tan(fov * 0.5)
    a = b / aspect
    c = (near + far) / (near - far)
    d = (2.0 * near * far) / (near - far)

    return np.array([[   a,  0.0,  0.0,  0.0 ], 
                     [ 0.0,    b,  0.0,  0.0 ], 
                     [ 0.0,  0.0,    c,    d ], 
                     [ 0.0,  0.0, -1.0,  0.0 ]])