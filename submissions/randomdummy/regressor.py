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
        
        return np.random.uniform(0,1, (X.shape[0],self.target_size) ) 
