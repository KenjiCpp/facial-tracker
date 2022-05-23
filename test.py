import numpy as np
import cv2
import pyrr
import math
from os.path import exists
from functools import partial
from utils import Candide3, FaceNormalizer, Eigenface, transform
from utils.OpenGL import *

mask = None

def main():
    fm = FaceNormalizer()
    ef = Eigenface()

    if exists('resources/eigenface_texture_generator.dat'):
        print('>>> Loading existing Eigenface Texture Generator...')
        ef.load('resources/eigenface_texture_generator.dat')
    else:
        print('>>> Creating and fitting new Eigenface Texture Generator...')
        cap = cv2.VideoCapture('resources/face.mp4')
        if (cap.isOpened()== False):
            print("Error opening video stream or file")
        normalize_faces = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            normalize_faces.append(fm.normalize(cv2.resize(frame, np.array(frame.shape[1::-1])) // 8).flatten())
        cap.release()
        ef.fit(normalize_faces)
        ef.save('resources/eigenface_texture_generator.dat')

    candide = Candide3()
    graphics = OpenGLGraphics()

    full_rect = OpenGLMesh([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]], [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], [[0, 1, 2], [2, 3, 0]])

    delta = np.zeros(candide.n_SUs)

    def dparams(params, d):
        res = np.array([params for _ in range(len(params))])
        res += np.identity(len(params)) * d
        return res

    def render(img, params, estimate, visualize=True, cal_diff=True):
        cnvs = OpenGLCanvas(img.shape[1], img.shape[0])
        cnvs.bind()

        a = params[0 : candide.n_AUs] * 0.001
        t = params[candide.n_AUs : candide.n_AUs + 3]
        r = params[candide.n_AUs + 3 : candide.n_AUs + 6]

        vertices, triangles = candide.generate_mesh(a, delta)
        trsn_mat = np.matmul(transform.perspective(math.pi / 4.0, img.shape[1] / img.shape[0], 10.0, 100.0), np.matmul(transform.translate(*t), transform.rotate(*r)))
        
        tex_coords = vertices.T
        tex_coords = np.matmul(trsn_mat, tex_coords)
        tex_coords = tex_coords.T

        model_norm = np.hstack([candide.vertices, np.ones(len(vertices)).reshape(-1, 1)])

        mesh = OpenGLMesh(model_norm, (tex_coords[:, :2] / tex_coords[:, 3:4] + 1.0) * 0.5, triangles)
        txtr = OpenGLTexture(data=img.astype(np.uint8), width=img.shape[1], height=img.shape[0])

        graphics.clear()
        graphics.draw(mesh, pyrr.Matrix44.identity(), txtr)

        wx = int(img.shape[1] * 0.2)
        mpd = cnvs.get_color_buffer()
        mpd = cv2.cvtColor(cv2.resize(mpd[0 : int(0.95 * img.shape[0]), wx : img.shape[1] - wx], (32, 32)), cv2.COLOR_BGR2GRAY)
        global mask
        if mask is None:
            mask = (mpd > 0.0).flatten()
        apx = np.zeros_like(estimate)
        apx[mask] = estimate[mask]
        rsd = (apx / np.max(apx)) - (mpd.flatten() / np.max(mpd))

        mesh.release()
        txtr.release()
        cnvs.release()

        tex_coords = ((tex_coords[:, :2] / tex_coords[:, 3:4] * np.array([1.0, -1.0]) + 1.0) * 0.5 * np.array(img.shape[1::-1])).astype(np.int32)

        vis = None
        if visualize:
            vis = img.copy()
            for t in triangles:
                vis = cv2.line(vis, tex_coords[t[0]], tex_coords[t[1]], (255, 255, 255), 1)
                vis = cv2.line(vis, tex_coords[t[0]], tex_coords[t[2]], (255, 255, 255), 1)
                vis = cv2.line(vis, tex_coords[t[2]], tex_coords[t[1]], (255, 255, 255), 1)

        dif = None
        if cal_diff:
            def diff(img, estimate, rsd, d, dpi):
                _, rsdi, __, ___ = render(img, dpi, estimate, False, False)
                drpi = (rsdi - rsd) / d
                return drpi
            d = -0.00002
            dp = dparams(params, d)
            dif = []
            for dpi in dp:
                dif.append(diff(img, estimate, rsd, d, dpi))
            dif = np.array(dif).T

        return mpd, rsd, vis, dif

    alpha     = np.zeros(candide.n_AUs)
    translate = np.array([0.55, 0.15, -2.3])
    rotate    = np.array([0.0, -0.2, -0.3])

    params = np.hstack([alpha, translate, rotate])

    cap = cv2.VideoCapture('resources/face.mp4')
    if cap.isOpened() == False:
        print("Error opening video stream or file")

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, np.array(frame.shape[1::-1]) // 8)
        
        iterations = 2
        for i in range(iterations):
            apr = ef.estimates[idx]
            mpd_show, rsd, vis_show, dif = render(frame, params, apr)
            rsd = rsd.reshape(-1, 1)
            rsd_show = rsd.reshape((32, 32))
            apr_show = apr.reshape((32, 32))
        
            U = np.linalg.pinv(dif)
            _dparams = np.matmul(U, rsd).flatten()
            params -= _dparams

        cv2.imshow("Visualization", cv2.resize(vis_show, np.array(vis_show.shape[1::-1]) * 3))
        cv2.imshow("Approximation", cv2.resize(apr_show, (300, 300)))
        cv2.imshow("Generated"    , cv2.resize(mpd_show, (300, 300)))
        cv2.imshow("Residual"     , cv2.resize(rsd_show, (300, 300)))

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        idx += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()