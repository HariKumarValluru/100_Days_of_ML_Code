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