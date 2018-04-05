import matplotlib.pyplot as plt
import numpy as np

# plt.plot([1, 2, 3, 4])
# plt.ylabel('some numbers')
# plt.show()
# line = 'fadsfdas, fdaf'
# line.rstrip()
# line.lstrip()

A = np.random.randn(4,3)
B = np.sum(A, axis = 1, keepdims = True)
print(B)
print(B.shape)

C = np.matrix('1 2; 3 4')
print(C)
print(C*C)
print(np.power(C, 2))

A = np.matrix('0.7 0.2 0.9 0.4 0.6')
print(A)
print(A.shape)

A = [0.7, 0.2, 0.5, 0.9, 0.1, 0.2]
A_T = [lambda x: 1 if x > 0.5 else 0 (x) for x in A]
print(list(A_T))

def fahrenheit(T):
    return ((float(9)/5)*T + 32)

temperatures = (36.5, 37, 37.5, 38, 39)
F = map(fahrenheit, temperatures)
print(list(F))
# print(A)
A_T = list(map(lambda x: 1 if x > 0.5 else 0, A))
print(list(A_T))
# X = range(10)
# print(X)
# print(type(X))
# # print((lambda x: 1 if x > 0.5 else 0)(0.5))
# a = [1,2,3,4]
# b = [17,12,11,10]
# res = map(lambda x,y:x+y, a,b)
