import numpy as np

class KNN():
    """ K Nearest Neighbors classifier.
    Parameters:
    -----------
    k: int
        The number of closest neighbors.
    """
    def __init__(self, k=5):
        self.k = k
    