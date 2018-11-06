# logistic regression
import numpy as np
from Utils.activation_functions import Sigmoid
from Utils.ml_utils import make_diagonal

class LogisticRegression:
    """ Logistic Regression classifier.
    Parameters:
    -----------
    learning_rate: float
    gradient_descent: boolean
    """
    def __init__(self, learning_rate=0.001):
        self.learning_rate = learning_rate
        self.params = None
        self.sigmoid = Sigmoid()
        
    def fit(self, X, y, iters =1000):
        n_features = X.shape[1]
        limit = 1 / np.sqrt(n_features)
        self.params = np.random.uniform(-limit, limit, (n_features))
        
        for i in range(iters):
            # using sigmoid activation function for making prediction
            y_pred = self.sigmoid(X.dot(self.params))            
            # gradient of the loss function
            self.params -= self.learning_rate * - (y - y_pred).dot(X)
        return
    
    def predict(self, X):
        y_pred =  np.round(self.sigmoid(X.dot(self.params))).astype(int)
        return y_pred