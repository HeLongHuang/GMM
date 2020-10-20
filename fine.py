import numpy as np
np.random.seed(1)
x = np.random.randint(1,5,size=[3,3,3])
y = np.random.randint(1,5,size=[3,3,3])

print(x)
print(y)
print(np.concatenate((x,y),axis=2))