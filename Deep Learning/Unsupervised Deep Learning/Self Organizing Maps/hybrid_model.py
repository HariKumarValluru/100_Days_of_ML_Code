# Hybrid deep learning model

# importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing the dataset
dataset = pd.read_csv("Datasets/Credit_Card_Applications.csv")

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))
X = sc.fit_transform(X)

# Training SOM
from minisom import MiniSom
som = MiniSom(x=10, y=10, input_len=15)
som.random_weights_init(X)
som.train_random(X, 100)

# Visualising the results
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

# finding the frauds
mappings = som.win_map(X)
frauds = np.concatenate((mappings[(2,8)], mappings[(5,4)]), axis=0)
frauds = sc.inverse_transform(frauds)

# creating the matrix of features
customers = dataset.iloc[:, 1:].values

# creating the dependent variables
is_fraud = np.zeros(customers.shape[0])

for i in range(0, len(dataset)):
    if dataset.iloc[i, 0] in frauds:
        is_fraud[i] = 1
        
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)

# ANN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(2, kernel_initializer = 'uniform', activation = 'relu',
                     input_dim = 15))

# Adding the output layer
classifier.add(Dense(1, kernel_initializer = 'uniform', 
                     activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', 
                   metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(customers, is_fraud, batch_size = 1, epochs = 5)

# Predicting the probabilities of frauds
y_pred = classifier.predict(customers)

y_pred = np.concatenate((dataset.iloc[:, 0:1].values, y_pred), axis=1)

y_pred = y_pred[y_pred[:, 1].argsort()]