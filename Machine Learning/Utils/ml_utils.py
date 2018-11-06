import numpy as np
import statsmodels.formula.api as sm

def train_test_split(X, y, test_size=0.25, shuffle=True, random_state=None):
    """ Split the data into train and test sets """
    if shuffle:
        X, y = shuffle_data(X, y, random_state)
    # Split the training data from test data in the ratio specified in
    # test_size
    split_i = len(y) - int(len(y) // (1 / test_size))
    X_train, X_test = X[:split_i], X[split_i:]
    y_train, y_test = y[:split_i], y[split_i:]

    return X_train, X_test, y_train, y_test

def shuffle_data(X, y, seed=None):
    """ Random shuffle of the samples in X and y """
    if seed:
        np.random.seed(seed)
    idx = np.arange(X.shape[0])
    np.random.permutation(idx)
    return X[idx], y[idx]

def backwardElimination(x, y, SL):
    """ Backward Elimination with p-values """
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    print(regressor_OLS.summary())
    return x

def correlation_matrix(X, Y=None):
    """ Calculate the correlation matrix for the dataset X """
    if Y is None:
        Y = X
    n_samples = np.shape(X)[0]
    covariance = (1 / n_samples) * (X - X.mean(0)).T.dot(Y - Y.mean(0))
    std_dev_X = np.expand_dims(std_dev(X), 1)
    std_dev_y = np.expand_dims(std_dev(Y), 1)
    correlation_matrix = np.divide(covariance, std_dev_X.dot(std_dev_y.T))

    return np.array(correlation_matrix, dtype=float)

def std_dev(X):
    """ Calculate the standard deviations of the features in dataset X """
    std_dev = np.sqrt(variance(X))
    return std_dev

def variance(X):
    """ Return the variance of the features in dataset X """
    mean = np.ones(np.shape(X)) * X.mean(0)
    n_samples = np.shape(X)[0]
    variance = (1 / n_samples) * np.diag((X - mean).T.dot(X - mean))
    return variance

def cost_function(X, Y, B):
    m = len(Y)
    J = np.sum((X.dot(B) - Y) ** 2)/(2 * m)
    return J

def gradient_descent(X, Y, B, alpha, iterations):
    cost_history = [0] * iterations
    m = len(Y)
    
    for iteration in range(iterations):
        # Hypothesis Values
        h = X.dot(B)
        # Difference b/w Hypothesis and Actual Y
        loss = h - Y
        # Gradient Calculation
        gradient = X.T.dot(loss) / m
        # Changing Values of B using Gradient
        B = B - alpha * gradient
        # New Cost Value
        cost = cost_function(X, Y, B)
        cost_history[iteration] = cost
        
    return B, cost_history

def rmse(Y, Y_pred):
    """Model Evaluation - RMSE (Root Mean Square Error)"""
    rmse = np.sqrt(sum((Y - Y_pred) ** 2) / len(Y))
    return rmse

def r2_score(Y, Y_pred):
    """"Model Evaluation - R2 Score"""
    mean_y = np.mean(Y)
    ss_tot = sum((Y - mean_y) ** 2)
    ss_res = sum((Y - Y_pred) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2

def cov_matrix(_y, _x):
    """Covariance between two vectors"""    
    if _x.shape[0] != _y.shape[0]:
        raise Exception("Shapes do not match")

    # make sure we use matrix multiplication, not array multiplication
    _xm = np.matrix(np.mean(_x, axis=0).repeat(_x.shape[0], axis = 0).reshape(_x.shape))
    _ym = np.matrix(np.mean(_y, axis=0).repeat(_y.shape[0], axis = 0).reshape(_y.shape))

    return ((_x - _xm).T * (_y - _ym)) * 1 / _x.shape[0]

def compute_b0_bn(ym, Xm):
    
    if ym.shape[1] != 1:
        raise Exception ("ym should be a vector with shape [n, 1]")
        
    if Xm.shape[0] != ym.shape[0]:
        raise Exception ("Xm should have the same amount of lines as ym")
    
    C_y_x = cov_matrix(ym, Xm)
    C_x_x = cov_matrix(Xm, Xm)

    b1_bn  = C_x_x.I * C_y_x
    
    x_mean  = np.matrix(np.mean(Xm, axis = 0))
    y_mean  = np.mean(ym)
    
    b0 = -x_mean * b1_bn + y_mean
    
    return (np.float(b0), np.array(b1_bn).flatten())

def accuracy_score(y_true, y_pred):
    """ Compare y_true to y_pred and return the accuracy """
    accuracy = np.sum(y_true == y_pred, axis=0) / len(y_true)
    return accuracy

def normalize(X, axis=-1, order=2):
    """ Scaling the dataset"""
    nm = np.linalg.norm(X, order, axis)
    # Convert inputs to arrays
    ld = np.atleast_1d(nm)
    ld[ld == 0] = 1
    
    return X / np.expand_dims(ld, axis)