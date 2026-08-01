import numpy as np

def mean_squareda_error(y,t):
    return 0.5*np.sum((y-t)**2)

def cross_entropy_error(y,t):  #需同时处理单个数据和批量数据#交叉熵误差函数
    if y.ndim==1:
        t=t.reshape(1,t.size)
        y=y.reshape(1,y.size)
    batch_size=y.shape[0]
    detla=1e-7    #添加一个微小值可以防止log值负无限大的发生
    return -np.sum(t*np.log(y+detla))/batch_size