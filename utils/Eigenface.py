import cv2
import pickle
import numpy as np
from sklearn.decomposition import PCA

class Eigenface:
    def __init__(self, n_components: int = 65, img_shape: tuple=(32, 32)):
        self.n_components = n_components
        self.img_shape = img_shape
        self.pca = PCA(n_components=n_components)

    def fit(self, img: list):
        xT = np.array(img)
        self.pca.fit(xT)
        self.eigenvectors = self.pca.components_.T
        self.eigenfaces   = self.eigenvectors.T.reshape((self.n_components, self.img_shape[0], self.img_shape[1]))
        self.mean         = self.pca.mean_.reshape(-1, 1)
        self.mean_img     = self.mean.flatten().reshape(self.img_shape)
        self.U            = np.matmul(np.linalg.inv(np.matmul(self.eigenvectors.T, self.eigenvectors)), self.eigenvectors.T)

        x  = xT.T
        dx = x - self.mean
        ep = np.matmul(self.U, dx)
        self.estimates = (np.matmul(self.eigenvectors, ep) + self.mean).T

    def generate(self, alpha: np.ndarray):
        return self.mean + np.sum(np.multiply(self.eigenfaces, alpha.reshape(-1, 1, 1)), axis=0)

    def save(self, file: str):
        save_obj = {
            'n_components' : self.n_components,
            'img_shape'    : self.img_shape,
            'mean'         : self.mean,
            'eigenvectors' : self.eigenvectors,
            'estimates'    : self.estimates
        }
        with open(file, 'wb') as f:
            pickle.dump(save_obj, f)
            

    def load(self, file: str):
        with open(file, 'rb') as f:
            save_obj = pickle.load(f)
        self.n_components = save_obj['n_components']
        self.img_shape    = save_obj['img_shape']
        self.mean         = save_obj['mean']
        self.eigenvectors = save_obj['eigenvectors']
        self.estimates    = save_obj['estimates']
        self.eigenfaces   = self.eigenvectors.T.reshape((self.n_components, self.img_shape[0], self.img_shape[1]))
        self.mean_img     = self.mean.flatten().reshape(self.img_shape)
        self.U            = np.matmul(np.linalg.inv(np.matmul(self.eigenvectors.T, self.eigenvectors)), self.eigenvectors.T)