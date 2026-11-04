import numpy as np

a = np.array([1,1,1,1,1])
std = a.std()
std += std == 0 
print(std)
