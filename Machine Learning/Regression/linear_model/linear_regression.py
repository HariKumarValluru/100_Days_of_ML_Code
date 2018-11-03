# Linear Regression Model
import numpy as np

class LinearRegression():
    
    def __init__(self):
        self.__a, self.__b = 0.0, 0.0
    
    def fit(self, x, y):
        mean_x, mean_y = np.mean(x), np.mean(y)
        r, sx, sy = self.corrcoeff(x, mean_x, y, mean_y)
        Sx = np.sqrt(sx / (np.size(x) - 1))
        Sy = np.sqrt(sy / (np.size(y) - 1))
        self.__b = self.slope(r, Sy, Sx)
        self.__a = self.intercept_(x, y, self.__b)
        return
    
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
        
        sx = self.variance(x, mean_x)
        sy = self.variance(y, mean_y)
        
        corrcoeff = covar / np.sqrt(sx * sy)
        
        return corrcoeff, sx, sy
    
    def slope(self, r, Sy, Sx):
        b = r * (Sy / Sx)
        return b
    
    def intercept_(self, x, y, b):
        mean_x, mean_y = np.mean(x), np.mean(y)
        a = mean_y - (b * mean_x)
        return a
    
    def predict(self, x):
        y_pred = []
        b0, b1 = self.__a, self.__b
        if np.size(x) > 1:
            for i in x:
                y_pred.append(b0 + (b1 * i))
        else:
            y_pred.append(b0 + (b1 * x))
        return np.array(y_pred)
       
        