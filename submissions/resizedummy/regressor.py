import numpy as np
from scipy.ndimage import zoom


def scale_up(x, size_factor):
    im = zoom(x, size_factor, order=2, mode='nearest')
    return im.reshape(-1)


class Regressor():
    """
    Dummy regressor
    Resizes the images by a factor of 2
    """
    def __init__(self):
        pass

    def fit(self, X, y):
        self.size_factor = np.sqrt(y.shape[1]).astype(int) // X.shape[1] ## 2

    def predict(self, X):
        y = np.zeros((X.shape[0], 128*128))
        for i in range(X.shape[0]):
            y[i] = scale_up(X[i], self.size_factor)
        return y # shape [N, 128*128]
