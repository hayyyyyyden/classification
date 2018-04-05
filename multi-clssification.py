import h5py
import numpy as np
import nn

with h5py.File('train_128.h5', 'r') as H: data = np.copy(H['data'])
with h5py.File('train_label.h5', 'r') as H: label = np.copy(H['label'])
with h5py.File('test_128.h5', 'r') as H: test = np.copy(H['data'])

# label = label.reshape(len(label), 1)
# print(data.shape)
# print(label.shape)
data1 = data[0:6000]
label1 = label[0:6000]
label1 = label1.reshape(len(label1), 1)



# res = nn.train(data, label, hiddenNum=2, unitNum=25, Thetas=None, precision=0.5)
# print(res)