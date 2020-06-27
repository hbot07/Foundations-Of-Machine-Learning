import numpy as np

x = np.array([1, 2, 3])
print(x ** 2)
l = x.size
x = x.reshape((1, x.size))
print(x)
print(np.concatenate((np.ones((1, l)), x)))
