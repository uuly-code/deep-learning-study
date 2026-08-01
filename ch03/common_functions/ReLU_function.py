import numpy as np
import matplotlib.pyplot as plt

def ReLU_func(x):
    return np.maximum(0,x)

x=np.arange(-5,6,1)
y=ReLU_func(x)

plt.plot(x,y,label="ReLU_func")
plt.xlabel("x")
plt.ylabel("y")
plt.title("ReLU_func")
plt.legend()
plt.show()