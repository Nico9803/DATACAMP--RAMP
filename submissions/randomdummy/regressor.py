import numpy as np


class Regressor():
    """
    Dummy regressor
    Gives random predictions
    """
    def __init__(self):
        pass

    def fit(self, X, y):
        self.target_size = y.shape[1]
        return

    def predict(self, X):
        N, size = X.shape
        return np.random.randint(0, 256, (N, self.target_size))

    def predict_proba(self, X):
        N, size = X.shape
        return np.random.randint(0, 256, (N, self.target_size))
