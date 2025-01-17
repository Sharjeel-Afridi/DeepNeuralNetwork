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

# aL, forw_cache = forward_propagation(X_train, params, 'relu')

# for l in range(len(params)//2 + 1):
#     print("Shape of A" + str(l) + " :", forw_cache['A' + str(l)].shape)


def compute_cost(AL, Y):

    m = Y.shape[1]

    if Y.shape[0] == 1:
        cost = -(1/m) * np.sum(Y*np.log(AL) + (1-Y)*np.log(1-AL))
    else:
        cost = -(1/m) * np.sum(Y.np.log(AL))
    
    cost = np.squeeze(cost)

    return cost

def backward_propagation(AL, Y, parameters, forward_cache, activation):

    grads = {}
    L = len(parameters)//2
    m = Y.shape[1]

    grads['dZ' + str(L)] = AL - Y
    grads['dW' + str(L)] = (1/m) * np.dot(grads["dZ" + str(L)], forward_cache['A' + str(L-1)].T)
    grads['db' + str(L)] = (1/m) * np.sum(grads["dZ" + str(L)], axis = 1, keepdims = True)

    for l in reversed(range(1,L)):
        if activation == 'tanh':
            grads["dZ" + str(l)] = np.dot(parameters['W' + str(l+1)].T,grads["dZ" + str(l+1)])*derivative_tanh(forward_cache['A' + str(l)])
        else:
            grads["dZ" + str(l)] = np.dot(parameters['W' + str(l+1)].T,grads["dZ" + str(l+1)])*derivative_relu(forward_cache['A' + str(l)])

        grads['dW' + str(l)] = (1/m) * np.dot(grads["dZ" + str(l)], forward_cache['A' + str(l-1)].T)
        grads['db' + str(l)] = (1/m) * np.sum(grads["dZ" + str(l)], axis = 1, keepdims = True)
    
    return grads
# grads = backward_propagation(forw_cache["A" + str(3)], Y_train, params, forw_cache, 'relu')

# for l in reversed(range(1, len(grads)//3 + 1)):
#     print("Shape of dZ" + str(l) + " :", grads['dZ' + str(l)].shape)
#     print("Shape of dW" + str(l) + " :", grads['dW' + str(l)].shape)
#     print("Shape of dB" + str(l) + " :", grads['db' + str(l)].shape, "\n")

def update_parameters(parameters, grads, learning_rate):

    L = len(parameters)//2

    for l in range(1, L):
        parameters["W" + str(l)] = parameters["W"+str(l)] - learning_rate*grads["dW"+ str(l)]
        parameters["b" + str(l)] = parameters["b"+str(l)] - learning_rate*grads["db"+ str(l)]
    return parameters


def model(X, Y, layer_dims, learning_rate, activation = 'relu', num_iterations = 100):

    parameters = initialize_parameters(layer_dims)

    for i in range(1, num_iterations):
        AL, forward_cache = forward_propagation(X, parameters, activation)
        
        cost = compute_cost(AL, Y)

        grads = backward_propagation(AL, Y, parameters, forward_cache, activation)

        parameters = update_parameters(parameters, grads, learning_rate)

        if i % (num_iterations/10) == 0:
            print("cost is", cost)
    return parameters

layers_dims = [X_train.shape[0], 20, 7, 5, Y_train.shape[0]] #  4-layer model
lr = 0.0075
iters = 2500

parameters = model(X_train, Y_train, layers_dims, learning_rate = lr, activation = 'relu', num_iterations = iters)