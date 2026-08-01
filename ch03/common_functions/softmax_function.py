import numpy as np

def softmax(A):
    c=np.max(A)     #防溢出
    exp_A=np.exp(A-c)
    Sum_exp_A=np.sum(exp_A)
    y=exp_A/Sum_exp_A
    return y


if __name__=="__main__":
    A=np.array([1,2,3])
    print(softmax(A))