# Restricted Boltzmann Machine

# importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# importing the dataset
movies = pd.read_csv('Datasets/ml-1m/movies.dat', sep='::', header=None, 
                     engine='python', encoding='latin-1')

users = pd.read_csv('Datasets/ml-1m/users.dat', sep='::', header=None, 
                     engine='python', encoding='latin-1')

ratings = pd.read_csv('Datasets/ml-1m/ratings.dat', sep='::', header=None, 
                     engine='python', encoding='latin-1')

# training and test set
training_set = pd.read_csv('Datasets/ml-100k/u1.base', delimiter='\t')
training_set = np.array(training_set, dtype="int")

test_set = pd.read_csv('Datasets/ml-100k/u1.test', delimiter='\t')
test_set = np.array(test_set, dtype="int")

# total number of users
tn_users = int(max(max(training_set[:,0]), max(test_set[:, 0])))
# total number of movies
tn_movies = int(max(max(training_set[:,1]), max(test_set[:, 1])))

# converting the data into array
def convert(data):
    new_data = []
    for id_users in range(1, tn_users + 1):
        id_movies = data[:, 1][data[:, 0] == id_users]
        id_ratings = data[:, 2][data[:, 0] == id_users]
        ratings = np.zeros(tn_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data

training_set = convert(training_set)
test_set = convert(test_set)

# converting the data into Torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# converting the ratings into binary ratings 1 (Liked) or 0 (Not Liked)
training_set[training_set == 0] = -1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1

test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1

# creating the architecture of the neural network
class RBM:
    def __init__(self, num_of_visible_nodes, num_of_hidden_nodes):
        # initialising the weights
        self.W = torch.randn(num_of_visible_nodes, num_of_hidden_nodes)
        # bias
        self.a = torch.randn(1, num_of_hidden_nodes)
        self.b = torch.randn(1, num_of_visible_nodes)
        
    def sample_h(self, x):
        """ sampling the hidden nodes """
        wx = torch.mm(x, self.W.t())
        activation = wx + self.a.expand_as(wx)
        # probability that the hidden node is activated given the value of the visible node
        P = torch.sigmoid(activation)
        return P, torch.bernoulli(P)
    
    def sample_v(self, y):
        """ sampling the visible nodes """
        wy = torch.mm(y, self.W)
        activation = wy + self.b.expand_as(wy)
        # probability that the visible node is activated given the value of the hidden node
        P = torch.sigmoid(activation)
        return P, torch.bernoulli(P)
    
    def train(self, v0, vk, ph0, phk):
        """
        training the model.
        Params:
            v0: input vector
            vk: visible nodes obtained after k samplings
            ph0: vector of probabilities
            phk: probabilities of the hidden nodes after k sampling.
        """
        self.W += torch.mm(v0.t(), ph0) - torch.mm(vk.t() - phk)
        self.b += torch.sum((v0 - vk), 0)
        self.a += torch.sum((ph0 - phk), 0)
        
nv = len(training_set[0]) # number of visible nodes
nh = 100 # number of hidden nodes
batch_size = 100

# initialising the model
rbm = RBM(num_of_visible_nodes = nv, num_of_hidden_nodes = nh)
