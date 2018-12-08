import numpy as np

class LinearRegression:
    
    def __init__(self):
        """
        Ordinary least squares Linear Regression.
        """
        self.mean_x = None
        self.mean_y = None
        self.b1 = None
        self.intercept_ = None
        
    def fit(self, x, y):
        """
        Fit linear model.
        Params:
            X : array-like or sparse matrix, shape (n_samples, n_features) 
                Training data
            y : array_like, shape (n_samples, n_targets) Target values.
        """
        # calculate the mean value of our x and y variables
        self.mean_x = np.mean(x)
        self.mean_y = np.mean(y)
        
        # Residuals of each x and y values from the means
        res_x = x - self.mean_x
        res_y = y - self.mean_y
        
        # Multiplication of the x and y residuals
        mul_res_xy = res_x * res_y
        
        # Summing the multiplied residuals
        sum_mul_xy = sum(mul_res_xy)
        
        # squared differences of each x
        sqrt_x = np.power(res_x, 2)
        
        # Summing the squared differences
        sum_sqrt_x = sum(sqrt_x)
        
        self.b1 = self.slope(sum_mul_xy, sum_sqrt_x)
        
        self.intercept_ = self.intercept(self.mean_x, self.mean_y)
        
        return self
    
    def predict(self, x):
        """
        Predict using the linear model
        Params:
            x : array_like or sparse matrix, shape (n_samples, n_features) 
                Samples.
        Returns:
            C : array, shape (n_samples,)
                Returns predicted values.
        """
        return self.intercept_ + self.b1 * x
        
    def slope(self, x1, x2):
        """
        Estimating The Slope
        Params:
            x1: Sum of multiplied residuals of x and y.
            x2: Sum of squared differences of x.
        """
        # calculate the value of slope
        return x1 / x2
    
    def intercept(self, mean_x, mean_y):
        """
        Estimating The Intercept
        Params:
            mean_x: mean of array-like or sparse matrix.
            mean_y: mean of array_like target values.
        """    
        return mean_y - self.b1 * mean_x