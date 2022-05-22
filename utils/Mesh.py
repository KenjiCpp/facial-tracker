import numpy as np
import cv2
import math

class Mesh:
    def __init__(self, vertices: list, triangles: list):
        self.vertices  = np.hstack([np.array(vertices), np.ones((len(vertices), 1))])
        self.triangles = triangles
        self.edges     = self.generate_edges()

    def generate_edges(self):
        def make_edge(f, t):
            if f < t:
                return (f, t)
            return (t, f)
        res = set()
        for triangle in self.triangles:
            res.add(make_edge(triangle[0], triangle[1]))
            res.add(make_edge(triangle[0], triangle[2]))
            res.add(make_edge(triangle[1], triangle[2]))
        return list(res)

    def draw(self, img: cv2.Mat, translation: np.ndarray, rotation: np.ndarray, z_range: tuple = (0.0, 100.0), fov: float = math.pi * 0.25):
        tfov = 1.0 / math.tan(fov / 2.0)
        aspc = img.shape[0] / img.shape[1]
        zrng = (z_range[1] + z_range[0]) / (z_range[1] - z_range[0])
        ztrs = -2.0 * z_range[1] * z_range[0] / (z_range[1] - z_range[0])
        projection = np.array([[ tfov * aspc,  0.0 ,  0.0 ,  0.0  ], 
                               [ 0.0        ,  tfov,  0.0 ,  0.0  ], 
                               [ 0.0        ,  0.0 ,  zrng,  ztrs ],
                               [ 0.0        ,  0.0 ,  1.0 ,  0.0  ]])
        MVP = np.matmul(projection, np.matmul(translation, rotation))

        p_obj = (self.vertices[:, :2] * np.array([1.0, -1.0]) + 1.0) * 0.5 * np.array([img.shape[1], img.shape[0]])
        p_cam = p_obj.copy()

        v = self.vertices.T
        v = np.matmul(MVP, v)
        v = v.T

        msh = img.copy()
        for e in self.edges:
            p1 = v[e[0]]
            p2 = v[e[1]]
            if p1[2] / p1[3] < -1.0 or p1[2] / p1[3] > 1.0 or p2[2] / p2[3] < -1.0 or p2[2] / p2[3] > 1.0:
                continue
            pt1 = (p1[0:2] / p1[3] * np.array([1.0, -1.0]) + 1.0) * 0.5 * np.array([img.shape[1], img.shape[0]])
            pt2 = (p2[0:2] / p2[3] * np.array([1.0, -1.0]) + 1.0) * 0.5 * np.array([img.shape[1], img.shape[0]])
            msh = cv2.line(msh, pt1.astype(np.int32), pt2.astype(np.int32), (255, 255, 255), 1)

        for i, p in enumerate(v):
            p_cam[i] = (p[0:2] / p[3] * np.array([1.0, -1.0]) + 1.0) * 0.5 * np.array([img.shape[1], img.shape[0]])
            if p[2] / p[3] < -1.0 or p[2] / p[3] > 1.0:
                continue
            pt = p_cam[i]
            msh = cv2.circle(msh, pt.astype(np.int32), 2, (255, 255, 255), 1)

        for t in self.triangles:
            a1 = p_cam[t[1]] - p_cam[t[0]]
            b1 = p_cam[t[2]] - p_cam[t[0]]
            a2 = p_obj[t[1]] - p_obj[t[0]]
            b2 = p_obj[t[2]] - p_obj[t[0]]
            M = np.matmul(np.array([a2, b2]).T, np.linalg.inv(np.array([a1, b1]).T))
            d = p_obj[t[0]].T - np.matmul(M, p_cam[t[0]].T)
            M = np.hstack([M, d.reshape(-1, 1)])

            mask = np.zeros_like(img)
            mask = cv2.fillConvexPoly(mask, np.array([p_obj[t[0]], p_obj[t[1]], p_obj[t[2]]]).astype(np.int32), (255, 255, 255)).astype(np.bool8)

            img2 = cv2.warpAffine(img, M, img.shape[1::-1])
            out = np.zeros_like(img)
            out[mask] = img2[mask]
    
        return msh, out
