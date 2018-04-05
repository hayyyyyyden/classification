import numpy as np

a = np.array([1,23,3]).reshape(1, -1)
print(np.shape(a))
a_trans = a.transpose()
print(np.shape(a_trans))
# print(a.T.shape)
# print(type(a))
np.matrix
