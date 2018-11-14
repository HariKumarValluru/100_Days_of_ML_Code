# Support Vector Machine

import numpy as np

class SVM:
    """The Support Vector Machine Model."""
    def __init__(self, c=1, kernel):
        return self
    
    def fit(self, X, y):
        n_samples, n_features = np.shape(X)
        
        
        pass
    
    def predict(self, X):
        # sign( w.x + b)
        y_pred = np.sign(np.dot(self.w, np.array(X)) + self.b)
        
        return y_pred