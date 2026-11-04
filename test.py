import numpy as np

a = np.arange(10).reshape((2,5))
b = np.array([1,2])


print(a.shape)
print(b.shape)
print(a+b)

a = a.reshape((5,2))
print(a+b)
