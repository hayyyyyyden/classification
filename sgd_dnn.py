# Package imports
from scipy.special import expit
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

    # A = 1 / (1 + np.exp(-Z))
    Z = np.clip(Z, -709, 36)
    A = expit(Z)
    if np.max(A) == 1:
        print('whats the fuck')
    cache = Z

    return A, cache


def softmax(Z):
    # 据说这样可以解决 overflow 的 bug，解决了个几把，垃圾
    # e_x = np.exp(Z - np.max(Z))
    # A = e_x / e_x.sum()
    Z = np.clip(Z, -709, 709)
    T = np.exp(Z)
    A = T / np.sum(T, axis=0)
    cache = Z
    return A, cache


def softmax_backward(dA, cache):
    # TODO: 什么鬼？为什么这里两步合成一步啊？
    dZ = dA
    return dZ


def relu_backward(dA, cache):
    """
    Implement the backward propagation for a single RELU unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """

    Z = cache[0]
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

    # s = 1 / (1 + np.exp(-Z))
    s = expit(Z)
    dZ = dA * s * (1 - s)

    assert (dZ.shape == Z.shape)

    return dZ


def initialize_parameters_deep(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network

    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """

    parameters = {}
    L = len(layer_dims)  # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

        assert (parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert (parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters


def initialize_parameters_he(layers_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the size of each layer.

    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
                    b1 -- bias vector of shape (layers_dims[1], 1)
                    ...
                    WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                    bL -- bias vector of shape (layers_dims[L], 1)
    """

    parameters = {}
    L = len(layers_dims) - 1  # integer representing the number of layers

    for l in range(1, L + 1):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * np.sqrt(
            2.0 / layers_dims[l - 1])
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))

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


def linear_activation_forward(A_prev, W, b, activation, keep_prob=1.0):
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
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    elif activation == "softmax":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = softmax(Z)
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
        D = np.random.rand(A.shape[0], A.shape[1])
        D = D <= keep_prob
        A = A * D
        A = A / keep_prob
        activation_cache = [activation_cache, D]

    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)
    return A, cache


def L_model_forward(X, parameters, keep_prob=1.0):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation

    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()

    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_activation_forward() (there are L-1 of them, indexed from 0 to L-1)
    """

    caches = []
    A = X
    L = len(parameters) // 2  # number of layers in the neural network

    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)],
                                             activation="relu", keep_prob=keep_prob)
        caches.append(cache)

    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    ### START CODE HERE ### (≈ 2 lines of code)
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation="softmax")
    caches.append(cache)
    ### END CODE HERE ###

    assert (AL.shape[1] == X.shape[1])

    return AL, caches


def L_model_backward_softmax(AL, Y, caches, lambd, keep_prob=1.0):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])

    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ...
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ...
    """
    grads = {}
    L = len(caches)  # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)  # after this line, Y is the same shape as AL

    # Initializing the backpropagation with softmax
    dAL = AL - Y  # 不需要除以 m 吗？为什么？

    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
    current_cache = caches[L - 1]
    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL,
                                                                                                      current_cache,
                                                                                                      "softmax", lambd=lambd)

    # Loop from l=L-2 to l=0
    for l in reversed(range(L - 1)):
        # lth layer: (RELU -> LINEAR) gradients.
        # Inputs: "grads["dA" + str(l + 1)], current_cache". Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)]
        ### START CODE HERE ### (approx. 5 lines)
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, "relu",
                                                                    lambd=lambd, keep_prob=keep_prob)
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
        ### END CODE HERE ###

    return grads


def compute_cost_softmax_with_reg(AL, parameters, lambd, Y):
    m = Y.shape[1]
    # Compute the L2 regularization.
    L = len(parameters) // 2  # number of layers in the neural network
    L2_regularization_cost = 0
    for l in range(1, L):
        Wl = parameters['W' + str(l)]
        L2_regularization_cost = L2_regularization_cost + np.sum(np.square(Wl))
    L2_regularization_cost = (lambd / (2 * m)) * L2_regularization_cost
    # Compute loss from aL and y.
    cc1 = np.log(AL)
    logprobs = np.multiply(cc1, Y)
    cost = (-1 / m) * np.nansum(logprobs)
    cost = cost + L2_regularization_cost
    cost = np.squeeze(cost)  # To make sure the shape is what we expect
    assert (cost.shape == ())
    return cost


def linear_backward(dZ, cache, lambd):
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

    dW = (1.0 / m) * np.dot(dZ, A_prev.T) + (lambd / m) * W
    db = (1.0 / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation, lambd, keep_prob=1.0):
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
        D = activation_cache[1]
        dA = dA * D
        dA = dA / keep_prob
        dZ = relu_backward(dA=dA, cache=activation_cache)
        dA_prev, dW, db = linear_backward(cache=linear_cache, dZ=dZ, lambd=lambd)

    elif activation == "softmax":
        dZ = softmax_backward(cache=activation_cache, dA=dA)
        dA_prev, dW, db = linear_backward(cache=linear_cache, dZ=dZ, lambd=lambd)

    elif activation == "sigmoid":
        dZ = sigmoid_backward(cache=activation_cache, dA=dA)
        dA_prev, dW, db = linear_backward(cache=linear_cache, dZ=dZ, lambd=lambd)

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
    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

    return parameters


# initialize_velocity

def initialize_velocity(parameters):
    """
    Initializes the velocity as a python dictionary with:
                - keys: "dW1", "db1", ..., "dWL", "dbL"
                - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
    Arguments:
    parameters -- python dictionary containing your parameters.
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl

    Returns:
    v -- python dictionary containing the current velocity.
                    v['dW' + str(l)] = velocity of dWl
                    v['db' + str(l)] = velocity of dbl
    """

    L = len(parameters) // 2  # number of layers in the neural networks
    v = {}

    # Initialize velocity
    for l in range(L):
        v["dW" + str(l + 1)] = np.zeros_like(parameters['W' + str(l + 1)])
        v["db" + str(l + 1)] = np.zeros_like(parameters['b' + str(l + 1)])

    return v


# update_parameters_with_momentum

def update_parameters_with_momentum(parameters, grads, v, beta, learning_rate):
    """
    Update parameters using Momentum

    Arguments:
    parameters -- python dictionary containing your parameters:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    grads -- python dictionary containing your gradients for each parameters:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
    v -- python dictionary containing the current velocity:
                    v['dW' + str(l)] = ...
                    v['db' + str(l)] = ...
    beta -- the momentum hyperparameter, scalar
    learning_rate -- the learning rate, scalar

    Returns:
    parameters -- python dictionary containing your updated parameters
    v -- python dictionary containing your updated velocities
    """

    L = len(parameters) // 2  # number of layers in the neural networks

    # Momentum update for each parameter
    for l in range(L):
        # compute velocities
        v["dW" + str(l + 1)] = v["dW" + str(l + 1)] * beta + (1 - beta) * grads['dW' + str(l + 1)]
        v["db" + str(l + 1)] = v["db" + str(l + 1)] * beta + (1 - beta) * grads['db' + str(l + 1)]
        # update parameters
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * v["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * v["db" + str(l + 1)]

    return parameters, v


# initialize_adam

def initialize_adam(parameters):
    """
    Initializes v and s as two python dictionaries with:
                - keys: "dW1", "db1", ..., "dWL", "dbL"
                - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.

    Arguments:
    parameters -- python dictionary containing your parameters.
                    parameters["W" + str(l)] = Wl
                    parameters["b" + str(l)] = bl

    Returns:
    v -- python dictionary that will contain the exponentially weighted average of the gradient.
                    v["dW" + str(l)] = ...
                    v["db" + str(l)] = ...
    s -- python dictionary that will contain the exponentially weighted average of the squared gradient.
                    s["dW" + str(l)] = ...
                    s["db" + str(l)] = ...

    """

    L = len(parameters) // 2  # number of layers in the neural networks
    v = {}
    s = {}

    # Initialize v, s. Input: "parameters". Outputs: "v, s".
    for l in range(L):
        v["dW" + str(l + 1)] = np.zeros_like(parameters["W" + str(l + 1)])
        v["db" + str(l + 1)] = np.zeros_like(parameters["b" + str(l + 1)])
        s["dW" + str(l + 1)] = np.zeros_like(parameters["W" + str(l + 1)])
        s["db" + str(l + 1)] = np.zeros_like(parameters["b" + str(l + 1)])

    return v, s


# update_parameters_with_adam

def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate=0.01,
                                beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    Update parameters using Adam

    Arguments:
    parameters -- python dictionary containing your parameters:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    grads -- python dictionary containing your gradients for each parameters:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
    v -- Adam variable, moving average of the first gradient, python dictionary
    s -- Adam variable, moving average of the squared gradient, python dictionary
    learning_rate -- the learning rate, scalar.
    beta1 -- Exponential decay hyperparameter for the first moment estimates
    beta2 -- Exponential decay hyperparameter for the second moment estimates
    epsilon -- hyperparameter preventing division by zero in Adam updates

    Returns:
    parameters -- python dictionary containing your updated parameters
    v -- Adam variable, moving average of the first gradient, python dictionary
    s -- Adam variable, moving average of the squared gradient, python dictionary
    """

    L = len(parameters) // 2  # number of layers in the neural networks
    v_corrected = {}  # Initializing first moment estimate, python dictionary
    s_corrected = {}  # Initializing second moment estimate, python dictionary

    # Perform Adam update on all parameters
    for l in range(L):
        # Moving average of the gradients. Inputs: "v, grads, beta1". Output: "v".
        v["dW" + str(l + 1)] = beta1 * v["dW" + str(l + 1)] + (1 - beta1) * grads['dW' + str(l + 1)]
        v["db" + str(l + 1)] = beta1 * v["db" + str(l + 1)] + (1 - beta1) * grads['db' + str(l + 1)]

        # Compute bias-corrected first moment estimate. Inputs: "v, beta1, t". Output: "v_corrected".
        v_corrected["dW" + str(l + 1)] = v["dW" + str(l + 1)] / (1 - beta1 ** t)
        v_corrected["db" + str(l + 1)] = v["db" + str(l + 1)] / (1 - beta1 ** t)

        # Moving average of the squared gradients. Inputs: "s, grads, beta2". Output: "s".
        s["dW" + str(l + 1)] = beta2 * s["dW" + str(l + 1)] + (1 - beta2) * np.square(grads['dW' + str(l + 1)])
        s["db" + str(l + 1)] = beta2 * s["db" + str(l + 1)] + (1 - beta2) * np.square(grads['db' + str(l + 1)])

        # Compute bias-corrected second raw moment estimate. Inputs: "s, beta2, t". Output: "s_corrected".
        s_corrected["dW" + str(l + 1)] = s["dW" + str(l + 1)] / (1 - beta2 ** t)
        s_corrected["db" + str(l + 1)] = s["db" + str(l + 1)] / (1 - beta2 ** t)

        # Update parameters. Inputs: "parameters, learning_rate, v_corrected, s_corrected, epsilon". Output: "parameters".
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * v_corrected["dW" + str(l + 1)] / (
                    np.sqrt(s_corrected["dW" + str(l + 1)]) + epsilon)
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * v_corrected["db" + str(l + 1)] / (
                    np.sqrt(s_corrected["db" + str(l + 1)]) + epsilon)

    return parameters, v, s



def L_layer_model_sgd(X, Y, layers_dims, optimizer, learning_rate=0.0007, mini_batch_size=64,
                      num_epochs=10000, beta=0.9, beta1=0.9, beta2=0.999, epsilon=1e-8,
                      keep_prob=1.0, lambd=50, print_cost=False):
    """
    L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SOFTMAX.

    Arguments:
    X -- input data, of shape (2, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    layers_dims -- python list, containing the size of each layer
    learning_rate -- the learning rate, scalar.
    mini_batch_size -- the size of a mini batch
    beta -- Momentum hyperparameter
    beta1 -- Exponential decay hyperparameter for the past gradients estimates
    beta2 -- Exponential decay hyperparameter for the past squared gradients estimates
    epsilon -- hyperparameter preventing division by zero in Adam updates
    num_epochs -- number of epochs
    print_cost -- True to print the cost every 1000 epochs

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    costs = []  # keep track of cost

    t = 0  # initializing the counter required for Adam update

    # Parameters initialization.
    # 使用He 初始化参数
    parameters = initialize_parameters_he(layers_dims)

    # Initialize the optimizer
    if optimizer == "gd":
        pass  # no initialization required for gradient descent
    elif optimizer == "momentum":
        v = initialize_velocity(parameters)
    elif optimizer == "adam":
        v, s = initialize_adam(parameters)

    m = Y.shape[1]

    # Loop (gradient descent)
    for i in range(num_epochs):
        # Define the random minibatches.
        minibatches = random_mini_batches(X, Y, mini_batch_size)

        for minibatch in minibatches:
            # Select a minibatch
            (minibatch_X, minibatch_Y) = minibatch

            # Forward propagation
            al, caches = L_model_forward(minibatch_X, parameters, keep_prob)

            # Compute cost with softmax.
            cost = compute_cost_softmax_with_reg(al, parameters, lambd, minibatch_Y)

            # Backward propagation with softmax.
            grads = L_model_backward_softmax(al, minibatch_Y, caches, lambd, keep_prob)

            # Update parameters
            if optimizer == "gd":
                parameters = update_parameters(parameters, grads, learning_rate)
            elif optimizer == "momentum":
                parameters, v = update_parameters_with_momentum(parameters, grads, v, beta, learning_rate)
            elif optimizer == "adam":
                t = t + 1  # Adam counter
                parameters, v, s = update_parameters_with_adam(parameters, grads, v, s,
                                                               t, learning_rate, beta1, beta2, epsilon)
        # Print the cost every 1000 epoch
        if print_cost and i % 10 == 0:
            print("Cost after epoch %i: %f" % (i, cost))
        if print_cost and i % 1 == 0:
            costs.append(cost)
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('epochs (per 100)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show(block=True)

    return parameters



# random_mini_batches
def random_mini_batches(X, Y, mini_batch_size=64):
    """
    Creates a list of random minibatches from (X, Y)

    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    mini_batch_size -- size of the mini-batches, integer

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """

    m = X.shape[1]  # number of training examples
    mini_batches = []

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((-1, m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(
        m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size: (k + 1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size: (k + 1) * mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size:]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


def predict(parameters: object, X: object) -> object:
    """
    Using the learned parameters, predicts a class for each example in X

    Arguments:
    parameters -- python dictionary containing your parameters
    X -- input data of size (n_x, m)

    Returns
    predictions -- vector of predictions of our model
    """

    AL, _ = L_model_forward(X, parameters)

    if AL.shape[1] > 1:
        temp = np.zeros((AL.shape[0], AL.shape[1]))
        predictions = np.argmax(AL, axis=0)
        for i in range(predictions.size):
            temp[predictions[i], i] = 1
        predictions = np.copy(temp)
    else:
        predictions = np.asarray(list(map(lambda x: 1 if x > 0.5 else 0, np.squeeze(np.asarray(AL))))).reshape(1, -1)

    return predictions


def gaussian_normalization(x, mean_x=None, sigma=None):
    if mean_x is None and sigma is None:
        mean_x = np.mean(x).reshape(-1, 1)
        sigma = np.std(x, axis=1, ddof=1).reshape(-1, 1)
        return (x - mean_x) / sigma, mean_x, sigma
    else:
        return (x - mean_x) / sigma


if __name__ == '__main__':
    # Shuffle the input data X and Y
    X = data
    m = X.shape[0]
    Y = label.reshape(m, 1)
    permutation = list(np.random.permutation(m))
    X = X[permutation, :]
    Y = Y[permutation, :].reshape(m, 1)
    print(X)
    print(X.shape)
    print(Y.shape)

    demo_size = 50000
    label_flag = 0
    X = data[0:demo_size].T
    Y = label[0:demo_size].reshape(demo_size, 1)
    Y = nn.adjustLabels(Y).T

    print(X)
    print(X.shape)
    print(Y.shape)

    # 给输入数据进行归一化
    X, mean_X, sigma = gaussian_normalization(X)

    # 检查数据的维度
    shape_X = X.shape
    shape_Y = Y.shape

    # 训练集的大小
    m = Y.shape[1]
    n_y = Y.shape[0]
    n_x = X.shape[0]

    print('矩阵 X 的大小是: ' + str(shape_X))
    print('矩阵 Y 的大小是: ' + str(shape_Y))
    print('我有 m = %d 个训练样本!' % m)

    # 开发集，应该叫 Dev
    test_right = 55000
    test_size = test_right - demo_size
    X_test = data[demo_size:test_right].T

    # 给测试数据进行归一化
    X_test = gaussian_normalization(X_test, mean_X, sigma)

    Y_test = label[demo_size:test_right].reshape(test_size, 1)
    Y_test = nn.adjustLabels(Y_test).T

    # 配置超参（各层节点数，迭代次数，学习速率）
    layers_dims = [n_x, 120, 50, 40, n_y]
    num_epochs = 2000
    learning_rate = 0.01
    optimizer = "adam"   # ["momentum", "gd," "adam"]
    mini_batch_size = 128
    beta = 0.9
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    keep_prob = 1.0
    # lambd = 0  # no regularization
    lambd = 0.02 * mini_batch_size
    print_cost = True

    # 训练
    parameters = L_layer_model_sgd(X, Y, layers_dims=layers_dims, optimizer=optimizer,
                                   learning_rate=learning_rate, mini_batch_size=mini_batch_size,
                                   num_epochs=num_epochs, beta=beta, beta1=beta1, beta2=beta2,
                                   epsilon=epsilon, keep_prob=keep_prob, lambd=lambd, print_cost=print_cost)
    # print('训练后的参数为' + str(parameters))

    # 计算训练集的拟合度
    predictions = predict(parameters, X)
    # 计算开发集的拟合度（test应该改为dev）
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
        accuracy_test = float(
            (np.dot(Y_test, prediction_test.T) + np.dot(1 - Y_test, 1 - prediction_test.T)) / float(Y_test.size) * 100)
    print("{}个隐层单元的训练集的拟合度为: {} %".format(str(layers_dims), accuracy))
    print('测试集的准确度为: %.4f ' % accuracy_test + '%')
