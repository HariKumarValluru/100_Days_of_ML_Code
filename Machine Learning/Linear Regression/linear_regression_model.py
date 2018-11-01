import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt

# Spliting the data into training and test sets
def train_test_split(dataset_x, dataset_y, split):
    row_count_x, row_count_y = dataset_x.shape[0], dataset_y.shape[0]
    x_split_point, y_split_point = int(row_count_x*0.3), int(row_count_y*0.3)
    x_test, x_train, y_test, y_train = dataset_x[:x_split_point], dataset_x[x_split_point:], dataset_y[:y_split_point], dataset_y[y_split_point:]
    return x_train, x_test, y_train, y_test

# Predicting the values
def predict(x, res):
    y_pred = []
    b0, b1 = res
    for i in x:
            y_pred.append(res[0] +  res[1] * i)
    return np.array(y_pred)

# Calculate covariance between x and y
def covariance(x, mean_x, y, mean_y):
	covar = 0.0
	for i in range(len(x)):
		covar += (x[i] - mean_x) * (y[i] - mean_y)
	return covar
 
# Calculate the variance of a list of numbers
def variance(values, mean):
	return sum([(x-mean)**2 for x in values])
 
# Calculate coefficients
def coefficients(x, y):
    x_mean, y_mean = np.mean(x), np.mean(y)
    b1 = covariance(x, x_mean, y, y_mean) / variance(x, x_mean)
    b0 = y_mean - b1 * x_mean
    return [b0, b1]

# Calculate root mean squared error
def rmse_metric(actual, predicted):
	sum_error = 0.0
	for i in range(len(actual)):
		prediction_error = predicted[i] - actual[i]
		sum_error += (prediction_error ** 2)
	mean_error = sum_error / float(len(actual))
	return sqrt(mean_error)

# load the data
customers = pd.read_csv("EcommerceCustomers.csv")

x = customers['Length of Membership'].values.reshape(-1,1)
y = customers['Yearly Amount Spent'].values.reshape(-1,1)

# Spliting the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, 0.8)

res = coefficients(x_train, y_train)

y_pred = predict(x_test, res)

# Visualising the training set results
plt.scatter(x_train, y_train, color="red")
plt.plot(x_train, predict(x_train, res), color="blue")
plt.title("Linear Regression (Training Set)")
plt.xlabel("Length of Membership")
plt.ylabel("Yearly Amount Spent")
plt.show()

# Visualising the test set results
plt.scatter(x_test, y_test, color="red")
plt.plot(x_train, predict(x_train, res), color="blue")
plt.title("Linear Regression (Test Set)")
plt.xlabel("Length of Membership")
plt.ylabel("Yearly Amount Spent")


# Calculating Root Mean Squared Error
print('RMSE: %.3f' % (rmse_metric(y_test, y_pred)))


import seaborn as sns
sns.distplot((y_test - y_pred), bins = 50)
pd.DataFrame(res, customers.iloc[:, 6:8].columns,columns = ['Coeffecient'])