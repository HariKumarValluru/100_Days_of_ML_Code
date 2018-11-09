# Collection of activation functions
import numpy as np

class Sigmoid():
    """ Outputs probability between 0 and 1, used to help define our logistic regression curve """
    """ Sigmoid Activation Function (1/1+e**-x)"""
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))