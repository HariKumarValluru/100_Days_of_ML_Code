# Collection of activation functions
import numpy as np

class Sigmoid():
    """ Sigmoid Activation Function (1/1+e**-x)"""
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))