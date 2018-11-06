# logistic regression
import numpy as np
from Utils.activation_functions import Sigmoid

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
        self.params = None
        self.sigmoid = Sigmoid()
        
    def fit(self, X, y, iters =1000):
        n_features = X.shape[1]
        limit = 1 / np.sqrt(n_features)
        self.params = np.random.uniform(-limit, limit, (n_features))
        
        for i in range(iters):
            # using sigmoid activation function for making prediction
            y_pred = self.sigmoid(X.dot(self.params))
            
        
        return y_pred
        