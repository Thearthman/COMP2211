from random import randint
import numpy as np
a = [randint(0,10) for _ in range(30)]
a = np.array(a).reshape(10,3)
b = np.random.randint(0,10,(2,3))
print(a)
print(b)

#a = a[:,None,:]
a = np.broadcast_to(a[:,None,:],(10,2,3))
print(a-b)
