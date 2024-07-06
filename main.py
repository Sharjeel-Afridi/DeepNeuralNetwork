import time
import random
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from keras.utils import to_categorical


X_train = np.loadtxt('datasets/cat_train_x.csv', delimiter = ',')/255.0
Y_train = np.loadtxt('datasets/cat_train_y.csv', delimiter = ',').reshape(1, X_train.shape[1])
X_test = np.loadtxt('datasets/cat_test_x.csv', delimiter = ',')/255.0
Y_test = np.loadtxt('datasets/cat_test_y.csv', delimiter = ',').reshape(1, X_test.shape[1])

# index = random.randrange(0, X_train.shape[1])
# plt.imshow(X_train[:, index].reshape(64,64, 3))
# plt.show()

def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    return A

def softmax(z):
    expZ = np.exp(z)
    return expZ/(np.sum(expZ, 0))

def relu(Z):
    A = np.maximum(0,Z)
    return A

def tanh(x):
    return np.tanh(x)

def derivative_relu(Z):
    return np.array(Z > 0, dtype = 'float')

def derivative_tanh(x):
    return (1 - np.power(x, 2))

# layer_dims = [X.shape[1], 100, 200, Y.shape[0]]

def initialize_parameters(layer_dims):

    L = len(layer_dims) - 1
    parameters = {}
    for l in range(1, L + 1):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
    
    return parameters

layer_dims = [X_train.shape[0], 100, 200, Y_train.shape[0]]
params = initialize_parameters(layer_dims)

for l in range(1, len(layer_dims)):
    print("Shape of W" + str(l) + ":", params['W' + str(l)].shape)
    print("Shape of B" + str(l) + ":", params['b' + str(l)].shape, "\n")

def forward_propagation(X, parameters, activation = 'relu'):

    forward_cache = {}
    L = len(parameters) // 2
    forward_cache['A0'] = X

    for l in range(1, L):

        forward_cache['Z' + str(l)] = parameters["W" + str(l)].dot(forward_cache["A"+ str(l - 1)]) + parameters['b'+str(l)]

        if activation == 'relu':
            forward_cache['A' + str(l)] = relu(forward_cache['Z' + str(l)])
        else:
            forward_cache['A' + str(l)] = tanh(forward_cache['Z' + str(l)])

    forward_cache['Z' + str(L)] = parameters["W" + str(L)].dot(forward_cache["A" + str(L-1)]) + parameters["b" + str(L)]

    if forward_cache["Z" + str(L)].shape[0] == 1:
        forward_cache['A' + str(L)] = sigmoid(forward_cache["Z" + str(L)])
    else:
        forward_cache['A' + str(L)] = softmax(forward_cache["Z" + str(L)])

    return forward_cache['A' + str(L)], forward_cache

aL, forw_cache = forward_propagation(X_train, params, 'relu')

for l in range(len(params)//2 + 1):
    print("Shape of A" + str(l) + " :", forw_cache['A' + str(l)].shape)
