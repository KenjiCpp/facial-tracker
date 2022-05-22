import cv2
import numpy as np
from parso import parse
from copy import deepcopy
from utils.Mesh import Mesh

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
        self.vertices = [Candide3.parse_vertex(line) for line in lines[idx : idx + self.n_vertices]]
        assert(len(self.vertices) == self.n_vertices)
        idx += self.n_vertices

        self.n_triangles = int(lines[idx])
        idx += 1
        self.triangles = [Candide3.parse_triangle(line) for line in lines[idx : idx + self.n_triangles]]
        assert(len(self.triangles) == self.n_triangles)
        idx += self.n_triangles

        self.n_AUs = int(lines[idx])
        idx += 1
        self.AUs = []
        for _ in range(self.n_AUs):
            n = int(lines[idx])
            idx += 1
            self.AUs.append([Candide3.parse_DU_element(line) for line in lines[idx : idx + n]])
            idx += n
        assert(len(self.AUs) == self.n_AUs)

        self.n_SUs = int(lines[idx])
        idx += 1
        self.SUs = []
        for _ in range(self.n_SUs):
            n = int(lines[idx])
            idx += 1
            self.SUs.append([Candide3.parse_DU_element(line) for line in lines[idx : idx + n]])
            idx += n
        assert(len(self.SUs) == self.n_SUs)

    def generate_mesh(self, alpha: np.ndarray, delta: np.ndarray):
        vertices = np.array(self.vertices)
        for a, au in zip(alpha, self.AUs):
            for aue in au:
                vertices[aue['idx']] += a * aue['offset']
        for d, su in zip(delta, self.SUs):
            for sue in su:
                vertices[sue['idx']] += d * sue['offset']
        return Mesh(vertices, self.triangles)

    def parse_vertex(line: str):
        v = line.split(' ')
        return np.array([float(v[0]), float(v[1]), float(v[2])])

    def parse_triangle(line: str):
        v = line.split(' ')
        return (int(v[0]), int(v[1]), int(v[2]))

    def parse_DU_element(line: str):
        v = line.split(' ')
        return {'idx': int(v[0]), 'offset': np.array([float(v[1]), float(v[2]), float(v[3])])}

