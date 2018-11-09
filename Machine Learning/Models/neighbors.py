import numpy as np

from Utils.ml_utils import euclidean_distance

class KNN():
    """ K Nearest Neighbors classifier.
    Parameters:
    -----------
    n_neighbors: int
        The number of closest neighbors (k value).
    """
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        return self
        
    def _vote(self, neighbor_labels):
        """ Return the most common class among the neighbors """
        counts = np.bincount(neighbor_labels.astype('int'))
        return counts.argmax()
    
    def predict(self, X_test):
        y_pred = np.empty(X_test.shape[0])
        # Determine the class of each sample
        for i, test_sample in enumerate(X_test):
            # Sort the training samples by their distance to the test sample and get the K nearest
            idx = np.argsort([euclidean_distance(test_sample, x) for x in self.X_train])[:self.n_neighbors]
            # Extract the labels of the K nearest neighboring training samples
            k_nearest_neighbors = np.array([self.y_train[i] for i in idx])
            # Label sample as the most common class label
            y_pred[i] = self._vote(k_nearest_neighbors) 

        return y_pred