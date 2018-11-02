# Linear Regression Model
import numpy as np

class LinearRegression():
    
    def __init__(self):
        pass
    
    def fit(self, x, y):
        mean_x, mean_y = np.mean(x), np.mean(y)
        r = self.corrcoeff(x, mean_x, y, mean_y)
        pass
    
    # Calculate covariance xy
    def covariance(self, x, mean_x, y, mean_y):
        covar = 0.0
        n = len(x)
        for i in range(n):
            covar += (x[i] - mean_x) * (y[i] - mean_y)
        return covar
    
    # Calculate the variance of a list of numbers
    def variance(self, values, mean):
        return sum([(x - mean)**2 for x in values ])
    
    def corrcoeff(self, x, mean_x, y, mean_y):
        covar = self.covariance(x, mean_x, y, mean_y)
        variance_x = self.variance(x, mean_x)
        variance_y = self.variance(y, mean_y)
        print(covar, variance_x, variance_y)
        pass
    
    def slope():
        pass
    
    def intercept_():
        pass
    
    def predict():
        pass
        