import cv2
import numpy as np
from parso import parse
from copy import deepcopy

class Candide3:
    def __init__(self):
        with open('resources/candide3.wfm','r') as f:
            lines = f.readlines()

        lines = [" ".join(line.strip('\n').split()) for line in lines]
        lines = [line for line in lines if len(line) > 0]
        lines = [line.strip() for line in lines if line[0] != '#']

        idx = 0

        self.n_vertices = int(lines[idx])
        idx += 1
        self.vertices = np.array([self.parse_vertex(line) for line in lines[idx : idx + self.n_vertices]])
        assert(len(self.vertices) == self.n_vertices)
        idx += self.n_vertices

        self.n_triangles = int(lines[idx])
        idx += 1
        self.triangles = [self.parse_triangle(line) for line in lines[idx : idx + self.n_triangles]]
        assert(len(self.triangles) == self.n_triangles)
        idx += self.n_triangles

        self.n_AUs = int(lines[idx])
        idx += 1
        self.AUs = []
        for _ in range(self.n_AUs):
            au = np.zeros_like(self.vertices)
            n = int(lines[idx])
            idx += 1
            for __ in range(n):
                v = lines[idx].split(' ')
                au[int(v[0])] += np.array([float(v[1]), float(v[2]), float(v[3])])
                idx += 1
            self.AUs.append(au)
        assert(len(self.AUs) == self.n_AUs)
        self.AUs = np.array(self.AUs)

        self.n_SUs = int(lines[idx])
        idx += 1
        self.SUs = []
        for _ in range(self.n_SUs):
            su = np.zeros_like(self.vertices)
            n = int(lines[idx])
            idx += 1
            for __ in range(n):
                v = lines[idx].split(' ')
                su[int(v[0])] += np.array([float(v[1]), float(v[2]), float(v[3])])
                idx += 1
            self.SUs.append(su)
        assert(len(self.SUs) == self.n_SUs)
        self.SUs = np.array(self.SUs)

    def generate_mesh(self, alpha: np.ndarray, delta: np.ndarray):
        deform = np.multiply(np.vstack([self.AUs, self.SUs]), np.hstack([alpha, delta]).reshape(-1, 1, 1))
        deform = np.sum(deform, axis=0)
        vertices = self.vertices + deform
        return np.hstack([vertices, np.ones(len(vertices)).reshape(-1, 1)]), self.triangles

    def parse_vertex(self, line: str):
        v = line.split(' ')
        return np.array([float(v[0]), float(v[1]), float(v[2])])

    def parse_triangle(self, line: str):
        v = line.split(' ')
        return (int(v[0]), int(v[1]), int(v[2]))


