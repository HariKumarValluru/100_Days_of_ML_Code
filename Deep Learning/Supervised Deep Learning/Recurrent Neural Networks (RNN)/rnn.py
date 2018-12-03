# Recurrent Neural Network

# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
dataset_train = pd.read_csv('Datasets/Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# reshaping the data
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# building the rnn
# importing the keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# initialising
regressor = Sequential()

# LSTM layer
regressor.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(50, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(50, return_sequences=True))
regressor.add(Dropout(0.2))

# no more LSTM layes
regressor.add(LSTM(50))
regressor.add(Dropout(0.2))

# output layer
regressor.add(Dense(1))

# compiling
regressor.compile(optimizer = 'adam', loss='mean_squared_error')

regressor.summary()

# fitting the train data
regressor.fit(X_train, y_train, epochs=100, batch_size=32)

regressor.save("google_stock_prediction.model")

# real stock price of January 2017
dataset_test = pd.read_csv("Datasets/Google_Stock_Price_Test.csv")
reals_stock_price = dataset_test.iloc[:, 1:2].values

# predicting the stock price of 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values

# reshaping the inputs
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)

X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
    
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_stock_price = regressor.predict(X_test)

predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising
plt.figure(figsize=(14,8))
# plotting the real stok price
plt.plot(reals_stock_price, color = 'r', label = "Real Google Stock Price")
# plotting the predicted stok price
plt.plot(predicted_stock_price, color = 'b', label = "Predicted Google Stock Price")
plt.title("Google Stock Price Prediction")
plt.xlabel("Month")
plt.ylabel("Stock Price")
plt.legend()
plt.show()