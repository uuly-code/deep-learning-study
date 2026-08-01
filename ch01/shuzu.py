import numpy as np 
x=np.array([[1,2],[23,46],[12,30],[9,11]])
y=np.array([0,2,4,6])
print(x)
x=x.flatten()
print(x)
print(x>20)
print(x[x>20])