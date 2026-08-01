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
y_out=H(0,0)
print(step_function(y_out))

#画图
step_vec=np.vectorize(step_function)
x_p=np.arange(-5.0,5.0,0.1)
y_p=step_vec(x_p)

plt.plot(x_p,y_p)
plt.xlabel("x")
plt.ylabel("y")
plt.title("activa_func")
plt.legend()
plt.show()
