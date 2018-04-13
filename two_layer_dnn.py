# Package imports
import numpy as np
import matplotlib.pyplot as plt
import scipy

from testCases_v2 import *
from planar_utils import plot_decision_boundary, load_planar_dataset, load_extra_datasets
import h5py
import nn
import math

with h5py.File('train_128.h5', 'r') as H: data = np.copy(H['data'])
with h5py.File('train_label.h5', 'r') as H: label = np.copy(H['label'])
with h5py.File('test_128.h5', 'r') as H: test = np.copy(H['data'])


def relu(Z):
    """
    Implement the RELU function.

    Arguments:
    Z -- Output of the linear layer, of any shape

    Returns:
    A -- Post-activation parameter, of the same shape as Z
    cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
    """

    A = np.maximum(0, Z)

    assert (A.shape == Z.shape)

    cache = Z
    return A, cache


def sigmoid(Z):
    """
    Implements the sigmoid activation in numpy

    Arguments:
    Z -- numpy array of any shape

    Returns:
    A -- output of sigmoid(z), same shape as Z
    cache -- returns Z as well, useful during backpropagation
    """

    A = 1 / (1 + np.exp(-Z))
    cache = Z

    return A, cache


def relu_backward(dA, cache):
    """
    Implement the backward propagation for a single RELU unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """

    Z = cache
    dZ = np.array(dA, copy=True)  # just converting dz to a correct object.

    # When z <= 0, you should set dz to 0 as well.
    dZ[Z <= 0] = 0

    assert (dZ.shape == Z.shape)

    return dZ


def sigmoid_backward(dA, cache):
    """
    Implement the backward propagation for a single SIGMOID unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """

    Z = cache

    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)

    assert (dZ.shape == Z.shape)

    return dZ


def initialize_parameters(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer

    Returns:
    parameters -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """


    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter
    cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """

    ### START CODE HERE ### (≈ 1 line of code)
    Z = np.dot(W, A) + b
    ### END CODE HERE ###

    assert (Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)

    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value
    cache -- a python dictionary containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """

    if activation == "sigmoid":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        ### START CODE HERE ### (≈ 2 lines of code)
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    #         print('==========This is sigmoid activation forward=========')
    #         print('cache A_prev = ' + str(linear_cache[0]))
    #         print('cache W = ' + str(linear_cache[1]))
    #         print('cache b = ' + str(linear_cache[2]))
    #         print('activation_cache Z = ' + str(activation_cache))
    #         print('==========This is sigmoid activation forward=========')
    ### END CODE HERE ###

    elif activation == "relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        ### START CODE HERE ### (≈ 2 lines of code)
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    #         print('==========This is relu activation forward=========')
    #         print('cache A_prev = ' + str(linear_cache[0]))
    #         print('cache W = ' + str(linear_cache[1]))
    #         print('cache b = ' + str(linear_cache[2]))
    #         print('activation_cache Z = ' + str(activation_cache))
    #         print('==========This is relu activation forward=========')
    ### END CODE HERE ###

    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)
    #     print('==========This is the linear_activation_forward =========')
    #     print('==The cache is a ((A,W,B),Z) structure.==')
    #     print('linear_activation_forward cache A_prev = ' + str(cache[0][0]))
    #     print('linear_activation_forward cache W = ' + str(cache[0][1]))
    #     print('linear_activation_forward cache b = ' + str(cache[0][2]))
    #     print('linear_activation_forward cache Z = ' + str(cache[1]))
    #     print('==The cache is a ((A,W,B),Z) structure.==')
    return A, cache


def compute_cost(AL, Y):
    """
    Implement the cost function defined by equation (7).

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """

    m = Y.shape[1]

    # Compute loss from aL and y.
    ### START CODE HERE ### (≈ 1 lines of code)
    logprobs = np.multiply(np.log(AL), Y) + np.multiply((1 - Y), np.log(1 - AL))
    cost = (-1 / m) * np.sum(logprobs)
    ### END CODE HERE ###

    cost = np.squeeze(cost)  # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert (cost.shape == ())

    return cost


def linear_backward(dZ, cache):
    """
    Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    # here the cache is not the caches from forward, but caches[l][0], which is a tuple of (A_prev, W, b). where caches[l][1] is cached Z.
    A_prev, W, b = cache

    m = A_prev.shape[1]

    ### START CODE HERE ### (≈ 3 lines of code)
    dW = (1.0 / m) * np.dot(dZ, A_prev.T)
    db = (1.0 / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    ### END CODE HERE ###

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.

    Arguments:
    dA -- post-activation gradient for current layer l
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache

    if activation == "relu":
        ### START CODE HERE ### (≈ 2 lines of code)
        dZ = relu_backward(dA=dA, cache=activation_cache)
        dA_prev, dW, db = linear_backward(cache=linear_cache, dZ=dZ)
        ### END CODE HERE ###

    elif activation == "sigmoid":
        ### START CODE HERE ### (≈ 2 lines of code)
        dZ = sigmoid_backward(cache=activation_cache, dA=dA)
        dA_prev, dW, db = linear_backward(cache=linear_cache, dZ=dZ)
        ### END CODE HERE ###

    return dA_prev, dW, db


def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent

    Arguments:
    parameters -- python dictionary containing your parameters
    grads -- python dictionary containing your gradients, output of L_model_backward

    Returns:
    parameters -- python dictionary containing your updated parameters
                  parameters["W" + str(l)] = ...
                  parameters["b" + str(l)] = ...
    """

    L = len(parameters) // 2  # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    ### START CODE HERE ### (≈ 3 lines of code)
    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
    ### END CODE HERE ###
    return parameters


def two_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):
    """
    Implements a two-layer neural network: LINEAR->RELU->LINEAR->SIGMOID.

    Arguments:
    X -- input data, of shape (n_x, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- dimensions of the layers (n_x, n_h, n_y)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- If set to True, this will print the cost every 100 iterations

    Returns:
    parameters -- a dictionary containing W1, W2, b1, and b2
    """

    np.random.seed(1)
    grads = {}
    costs = []  # to keep track of the cost
    m = X.shape[1]  # number of examples
    (n_x, n_h, n_y) = layers_dims


    # Initialize parameters dictionary, by calling one of the functions you'd previously implemented
    ### START CODE HERE ### (≈ 1 line of code)
    n_x, n_h, n_y = layers_dims
    parameters = initialize_parameters(n_x, n_h, n_y)
    ### END CODE HERE ###

    # Get W1, b1, W2 and b2 from the dictionary parameters.
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # Loop (gradient descent)

    for i in range(0, num_iterations):

        # Forward propagation: LINEAR -> RELU -> LINEAR -> SIGMOID. Inputs: "X, W1, b1, W2, b2". Output: "A1, cache1, A2, cache2".
        ### START CODE HERE ### (≈ 2 lines of code)
        A1, cache1 = linear_activation_forward(X, W1, b1, activation="relu")
        A2, cache2 = linear_activation_forward(A1, W2, b2, activation="sigmoid")
        ### END CODE HERE ###

        # Compute cost
        ### START CODE HERE ### (≈ 1 line of code)
        cost = compute_cost(A2, Y)
        ### END CODE HERE ###

        # Initializing backward propagation
        dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))

        # Backward propagation. Inputs: "dA2, cache2, cache1". Outputs: "dA1, dW2, db2; also dA0 (not used), dW1, db1".
        ### START CODE HERE ### (≈ 2 lines of code)
        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, activation="sigmoid")
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, activation="relu")
        ### END CODE HERE ###

        # Set grads['dWl'] to dW1, grads['db1'] to db1, grads['dW2'] to dW2, grads['db2'] to db2
        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2

        # Update parameters.
        ### START CODE HERE ### (approx. 1 line of code)
        parameters = update_parameters(parameters, grads, learning_rate)
        ### END CODE HERE ###

        # Retrieve W1, b1, W2, b2 from parameters
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]

        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    # plot the cost

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    plt.show(block=False)

    return parameters


def predict(parameters: object, X: object) -> object:
    """
    Using the learned parameters, predicts a class for each example in X

    Arguments:
    parameters -- python dictionary containing your parameters
    X -- input data of size (n_x, m)

    Returns
    predictions -- vector of predictions of our model
    """

    # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    A1, _ = linear_activation_forward(X, W1, b1, activation="relu")
    A2, _ = linear_activation_forward(A1, W2, b2, activation="sigmoid")

    if A2.shape[1] > 1:
        temp = np.zeros((A2.shape[0],A2.shape[1]))
        predictions = np.argmax(A2, axis=0)
        for i in range(predictions.size):
            temp[predictions[i], i] = 1
        predictions = np.copy(temp)
    else:
        predictions = np.asarray(list(map(lambda x: 1 if x > 0.5 else 0, np.squeeze(np.asarray(A2))))).reshape(1, -1)

    return predictions


if __name__ == '__main__':
    demo_size = 20000
    label_flag = 0
    X = data[0:demo_size].T
    Y = label[0:demo_size].reshape(demo_size, 1)
    Y = nn.adjustLabels(Y).T
    # Y = Y[label_flag, :].reshape(1, demo_size)

    print(X)
    print(X.shape)
    print(Y.shape)

    # 检查数据的维度
    shape_X = X.shape
    shape_Y = Y.shape
    m = Y.shape[1]  # 训练集的大小
    n_y = Y.shape[0]
    n_x = X.shape[0]

    print('矩阵 X 的大小是: ' + str(shape_X))
    print('矩阵 Y 的大小是: ' + str(shape_Y))
    print('我有 m = %d 个训练样本!' % m)

    # 测试不同参数的准确度
    hidden_layer_sizes = [1, 2, 3, 4, 5, 10, 20, 50]
    hidden_layer_sizes = [50]

    # 预测测试集的准确度
    test_right = 30000
    test_size = test_right - demo_size
    X_test = data[demo_size:test_right].T
    Y_test = label[demo_size:test_right].reshape(test_size, 1)
    Y_test = nn.adjustLabels(Y_test).T
    # Y_test = Y_test[label_flag, :].reshape(1, test_size)

    for _, n_h in enumerate(hidden_layer_sizes):
        # 迭代次数
        parameters = two_layer_model(X, Y, layers_dims=(n_x, n_h, n_y), num_iterations=3700, learning_rate=0.002,
                                     print_cost=True)
        # print('训练后的参数为' + str(parameters))
        # 计算训练集的拟合度
        predictions = predict(parameters, X)
        prediction_test = predict(parameters, X_test)
        if n_y > 1:
            temp = np.dot(Y.T, predictions)
            correction_num = sum(temp[i][i] for i in range(Y.shape[1]))
            accuracy = float(correction_num) / float(Y.shape[1]) * 100
            temp_test = np.dot(Y_test.T, prediction_test)
            correction_num_test = sum(temp_test[i][i] for i in range(Y_test.shape[1]))
            accuracy_test = float(correction_num_test) / float(Y_test.shape[1]) * 100
        else:
            accuracy = float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100)
            accuracy_test = float((np.dot(Y_test, prediction_test.T) + np.dot(1 - Y_test, 1 - prediction_test.T)) / float(Y_test.size) * 100)
        print("{}个隐层单元的训练集的拟合度为: {} %".format(n_h, accuracy))
        print('测试集的准确度为: %.4f' % accuracy_test)


