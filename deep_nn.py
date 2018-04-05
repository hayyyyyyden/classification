# Package imports
import numpy as np
import matplotlib.pyplot as plt
import scipy

from testCases_v2 import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets
import h5py
import nn
import math

with h5py.File('train_128.h5', 'r') as H: data = np.copy(H['data'])
with h5py.File('train_label.h5', 'r') as H: label = np.copy(H['label'])
with h5py.File('test_128.h5', 'r') as H: test = np.copy(H['data'])

np.random.seed(1) # set a seed so that the results are consistent


# layer_sizes 得到神经网络各层的大小
def layer_sizes(X, Y):
    """
    Arguments:
    X -- input dataset of shape (input size, number of examples)
    Y -- labels of shape (output size, number of examples)

    Returns:
    n_x -- the size of the input layer
    n_y -- the size of the output layer
    """

    n_x = X.shape[0]  # size of input layer
    n_y = Y.shape[0]  # size of output layer

    return n_x, n_y


# initialize_parameters 初始化参数
def initialize_parameters(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer

    Returns:
    params -- python dictionary containing your parameters:
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


# forward_propagation 向前传播
def forward_propagation(X, parameters):
    """
    Argument:
    X -- input data of size (n_x, m)
    parameters -- python dictionary containing your parameters (output of initialization function)

    Returns:
    A2 -- The sigmoid output of the second activation
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
    """
    # Retrieve each parameter from the dictionary "parameters"
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # Implement Forward Propagation to calculate A2 (probabilities)
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    assert (A2.shape[1] == X.shape[1])

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}

    return A2, cache


# compute_cost 计算误差
def compute_cost(A2, Y, parameters):
    """
    Computes the cross-entropy cost given in equation (13)

    Arguments:
    A2 -- The sigmoid output of the second activation, of shape (1, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    parameters -- python dictionary containing your parameters W1, b1, W2 and b2

    Returns:
    cost -- cross-entropy cost given equation (13)
    """

    m = Y.shape[1]  # number of example

    # Compute the cross-entropy cost
    logprobs = np.multiply(np.log(A2), Y) + np.multiply((1 - Y), np.log(1 - A2))
    cost = (-1 / m) * np.sum(logprobs)

    cost = np.squeeze(cost)  # makes sure cost is the dimension we expect.
    # E.g., turns [[17]] into 17
    assert (isinstance(cost, float))

    return cost


def thanh_derivative(x):
    """thanH求导
    """
    a = np.tanh(x)
    return 1 - np.power(a, 2)


# backward_propagation 向后传播
def backward_propagation(parameters, cache, X, Y):
    """
    Implement the backward propagation using the instructions above.

    Arguments:
    parameters -- python dictionary containing our parameters
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
    X -- input data of shape (2, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)

    Returns:
    grads -- python dictionary containing your gradients with respect to different parameters
    """
    m = X.shape[1]

    # First, retrieve W1 and W2 from the dictionary "parameters".
    W1 = parameters["W1"]
    W2 = parameters["W2"]

    # Retrieve also A1 and A2 from dictionary "cache".
    A1 = cache["A1"]
    A2 = cache["A2"]
    Z1 = cache["Z1"]

    # Backward propagation: calculate dW1, db1, dW2, db2.
    dZ2 = A2 - Y
    dW2 = (1.0 / m) * np.dot(dZ2, A1.T)
    db2 = (1.0 / m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.multiply(np.dot(W2.T, dZ2), thanh_derivative(Z1))
    dW1 = (1.0 / m) * np.dot(dZ1, X.T)
    db1 = (1.0 / m) * np.sum(dZ1, axis=1, keepdims=True)

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}

    return grads


# update_parameters 更新参数
def update_parameters(parameters, grads, learning_rate=1.2):
    """
    Updates parameters using the gradient descent update rule given above

    Arguments:
    parameters -- python dictionary containing your parameters
    grads -- python dictionary containing your gradients

    Returns:
    parameters -- python dictionary containing your updated parameters
    """
    # Retrieve each parameter from the dictionary "parameters"
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # Retrieve each gradient from the dictionary "grads"
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    # Update rule for each parameter
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


# nn_model 神经网络模型
def nn_model(X, Y, n_h, num_iterations=10000, print_cost=False, learning_rate=0.012):
    """
    Arguments:
    X -- dataset of shape (2, number of examples)
    Y -- labels of shape (1, number of examples)
    n_h -- size of the hidden layer
    num_iterations -- Number of iterations in gradient descent loop
    print_cost -- if True, print the cost every 1000 iterations

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    n_x, n_y = layer_sizes(X, Y)

    # Initialize parameters, then retrieve W1, b1, W2, b2. Inputs: "n_x, n_h, n_y". Outputs = "W1, b1, W2, b2, parameters".
    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # Loop (gradient descent)

    for i in range(0, num_iterations):

        # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".
        A2, cache = forward_propagation(X, parameters)

        # Cost function. Inputs: "A2, Y, parameters". Outputs: "cost".
        cost = compute_cost(A2, Y, parameters)

        # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
        grads = backward_propagation(parameters, cache, X, Y)

        # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".
        parameters = update_parameters(parameters, grads, learning_rate)

        # Print the cost every 1000 iterations
        if (print_cost and i % 100 == 0) or i == num_iterations - 1:
            print("第 %i 次迭代后的误差为: %f" % (i, cost))

    return parameters


# predict 预测函数

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
    A2, _ = forward_propagation(X, parameters)
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
    # data1 = data[0:6000]
    # label1 = label[0:6000]
    # label1 = label1.reshape(len(label1), 1)
    demo_size = 20000
    label_flag = 0
    X = data[0:demo_size].T
    # X = X[0:2, :]
    Y = label[0:demo_size].reshape(demo_size, 1)
    Y = nn.adjustLabels(Y).T
    # Y 的 shape 是(10,demo_size)
    # Y = Y[label_flag, :].reshape(1, demo_size)

    # X, Y = load_planar_dataset()
    print(X)
    print(X.shape)
    print(Y.shape)
    # 数据可视化:
    # plt.scatter(X[0, :], X[1, :], c=Y.reshape(demo_size,), s=math.sqrt(demo_size), cmap=plt.cm.Spectral)
    # plt.show(block=False)

    # 检查数据的维度
    shape_X = X.shape
    shape_Y = Y.shape
    m = Y.shape[1]  # 训练集的大小

    print('矩阵 X 的大小是: ' + str(shape_X))
    print('矩阵 Y 的大小是: ' + str(shape_Y))
    print('我有 m = %d 个训练样本!' % m)

    (n_x, n_y) = layer_sizes(X, Y)
    # the size of hidden layer
    n_h = 4
    print("The size of the input layer is: n_x = " + str(n_x))
    print("The size of the hidden layer is: n_h = " + str(n_h))
    print("The size of the output layer is: n_y = " + str(n_y))

    parameters = initialize_parameters(n_x, n_h, n_y)
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))

    # 向前传播，预测
    A2, cache = forward_propagation(X, parameters)

    print('Z1 ' + str(cache['Z1'].shape))
    print('A1 ' + str(cache['A1'].shape))
    print('Z2 ' + str(cache['Z2'].shape))
    print('A2 ' + str(cache['A2'].shape))

    # 计算误差
    print("cost = " + str(compute_cost(A2, Y, parameters)))

    # 先后传播，计算梯度
    grads = backward_propagation(parameters, cache, X, Y)
    print("dW1 = " + str(grads["dW1"]))
    print("db1 = " + str(grads["db1"]))
    print("dW2 = " + str(grads["dW2"]))
    print("db2 = " + str(grads["db2"]))

    parameters = update_parameters(parameters, grads)

    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))

    A2, cache = forward_propagation(X, parameters)

    # 计算误差
    print("cost = " + str(compute_cost(A2, Y, parameters)))

    # 测试不同参数的准确度
    hidden_layer_sizes = [1, 2, 3, 4, 5, 10, 20, 50]
    hidden_layer_sizes = [15]

    # 预测测试集的准确度
    test_right = 30000
    test_size = test_right - demo_size
    X_test = data[demo_size:test_right].T
    Y_test = label[demo_size:test_right].reshape(test_size, 1)
    Y_test = nn.adjustLabels(Y_test).T
    # Y_test = Y_test[label_flag, :].reshape(1, test_size)

    for _, n_h in enumerate(hidden_layer_sizes):
        # 迭代次数
        parameters = nn_model(X, Y, n_h, num_iterations=10000, print_cost=True, learning_rate=0.2)
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


    # Plot the decision boundary
    # plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
    # plt.title("Decision Boundary for hidden layer size " + str(4))
    # plt.show(block=True)

    # 训练逻辑斯谛回归分类器
    # clf = sklearn.linear_model.LogisticRegressionCV()
    # clf = sklearn.linear_model.SGDClassifier(loss='log')
    # clf.fit(X.T, np.ravel(Y.T))

    # 使用 scipy.sparse.matrix 来解决内存问题，不成功
    # X = scipy.sparse.csc_matrix(X)
    # Y = scipy.sparse.csc_matrix(np.ravel(Y))

    # 绘制逻辑斯谛回归下的决策边界
    # print('what?')
    # plot_decision_boundary(lambda x: clf.predict_proba(x), X, Y)
    # plt.title("Logistic Regression")
    # plt.show(block=False)
    # print('what?')

    # 打印准确度
    # LR_predictions = clf.predict(X.T)
    # print('what?')
    # print(LR_predictions.shape)
    # print((1 - LR_predictions).shape)
    # print(Y.shape)
    # print((1 - Y).shape)
    # print(np.dot(1 - Y, 1 - LR_predictions))
    # print(np.dot(Y, LR_predictions))
    # print((102 + 86) / 400)
    # print('Accuracy of logistic regression: %d ' % float(
    #     (np.dot(Y, LR_predictions) + np.dot(1 - Y, 1 - LR_predictions)) / float(Y.size) * 100) +
    #       '% ' + "(percentage of correctly labelled data points)")



