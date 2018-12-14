# AutoEncoders

# importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# Importing the dataset
movies = pd.read_csv('Datasets/ml-1m/movies.dat', sep='::', header=None, 
                     engine='python', encoding='latin-1')

users = pd.read_csv('Datasets/ml-1m/users.dat', sep='::', header=None, 
                     engine='python', encoding='latin-1')

ratings = pd.read_csv('Datasets/ml-1m/ratings.dat', sep='::', header=None, 
                     engine='python', encoding='latin-1')

# loading training and test set
training_set = pd.read_csv('Datasets/ml-100k/u1.base', delimiter='\t', header=None)
test_set = pd.read_csv('Datasets/ml-100k/u1.test', delimiter='\t', header=None)

# total number of users
nb_users = int(max(training_set[0].max(), test_set[0].max()))
# total number of movies
nb_movies = int(max(training_set[1].max(), test_set[1].max()))

# converting the data into a matrix with users in lines and movies in columns
training_set = training_set.pivot_table(index=0, columns=1, values=2)
test_set = test_set.pivot_table(index=0, columns=1, values=2)

# fill the missing values
training_set = training_set.fillna(0)
test_set = test_set.fillna(0)

# converting the data into Torch tensors
training_set = torch.FloatTensor(training_set.values)
test_set = torch.FloatTensor(test_set.values)

# Creating the architecture of the autoencoders
class SAE(nn.Module):
    """Stacked Auto Encoder"""
    def __init__(self, ):
        super(SAE, self).__init__()
        # encoding layers
        self.fc1 = nn.Linear(nb_movies, 20)
        self.fc2 = nn.Linear(20, 10)
        # decoding layers
        self.fc3 = nn.Linear(10, 20)
        self.fc3 = nn.Linear(20, nb_movies)
        # activation function
        self.activation = nn.Sigmoid()
    
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x

        
        