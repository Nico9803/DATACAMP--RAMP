import numpy as np
from scipy.ndimage import zoom


def scale_up(x, size_factor):
    size = np.sqrt(x.shape[0]).astype(int)
    im = x.reshape(size, size)
    im = zoom(im, size_factor, order=3).astype(int)
    return im.reshape(-1)


class Regressor():
    """
    Dummy regressor
    Resizes the images by a factor of 5
    """
    def __init__(self):
        pass

    def fit(self, X, y):
        self.size_factor = np.sqrt(y.shape[1] / X.shape[1])

    def predict(self, X):
        return np.apply_along_axis(lambda x: scale_up(x, self.size_factor), 1,
                                   X)

    def predict_proba(self, X):
        return np.apply_along_axis(scale_up, 1, X)
