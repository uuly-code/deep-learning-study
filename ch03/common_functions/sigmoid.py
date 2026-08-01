import numpy as np
import matplotlib.pyplot as plt

def H(x1,x2):
    x=np.array([x1,x2])
    w=np.array([0.5,0.5])
    b=-0.7
    y=b+np.sum(x*w)
    return y
def step_function(y):
    if y>0:
        return 1
    else:
        return 0


def sigmoid(x):
    return 1/(1+np.exp(-x))

# x=np.arange(-5.0,5.0,0.1)
# y1=sigmoid(x)
# step_vec=np.vectorize(step_function)
# y2=step_vec(x)

# plt.plot(x,y1,label='sigmoid_func')
# plt.plot(x,y2,linestyle="--",label='step_func')
# plt.xlabel("x")
# plt.ylabel("y")
# plt.title("sigmoid_func&step_func")
# plt.legend()
# plt.show()