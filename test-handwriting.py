# coding: utf-8
# neural_network/test_handwritten_digits.py
"""手写字符集
"""
import nn
import numpy as np
from scipy.io import loadmat
import h5py

with h5py.File('train_128.h5', 'r') as H: data = np.copy(H['data'])
with h5py.File('train_label.h5', 'r') as H: label = np.copy(H['label'])
with h5py.File('test_128.h5', 'r') as H: test = np.copy(H['data'])

# data = loadmat('handwritten_digits.mat')
# Thetas = loadmat('ex4weights.mat')
# Thetas_p = [Thetas['Theta1'], Thetas['Theta2']]
# print(Thetas_p)
# dim1 = len(Thetas_p)
# dim2 = len(Thetas_p[0][0])
# dim3 = len(Thetas_p[1][0])
# print(dim1,dim2,dim3)
# print(np.array(Thetas_p).shape)

# X = np.mat(data['X'])
# y = np.mat(data['y'])

# print(y[1750:1800,:])

# data = data[0:2000, :]
# label_train = label[0:2000]
#
# print(data.shape)
# print(label_train.shape)
#
#
# print('+++++++++')
test = np.mat(test[50000:60000, :])
label_t = np.mat(label[50000:60000]).transpose()
# print(test.shape)
# print(label_t.shape)
#
X = np.mat(data)
y = np.mat(label).transpose()
#
# print(X.shape)
# print(y[1490:1500,:])
#
res = nn.train(X, y, hiddenNum=1, unitNum=25, Thetas=None, precision=0.5)
print(res['Thetas'])
theta = res['Thetas']
predict = nn.predict(test, theta)
print('预测结果')
print(predict)
print(predict.shape)
print(predict[1:10])
print(label_t[1:10])
#
#
# print('Error is: %.4f'%res['error'])