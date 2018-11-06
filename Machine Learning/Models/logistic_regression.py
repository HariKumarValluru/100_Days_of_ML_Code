# logistic regression
import numpy as np

class LogisticRegression:
    """ Logistic Regression classifier.
    Parameters:
    -----------
    learning_rate: float
    gradient_descent: boolean
    """
    def __init__(self, learning_rate=0.01, gradient_descent=True):
        self.learning_rate = learning_rate
        self.gradient_descent = gradient_descent
        
    def fit(self, X, y, iters =1000):
        n_features = X.shape[1]
        limit = 1 / np.sqrt(n_features)
        
        return n_features, limit
        