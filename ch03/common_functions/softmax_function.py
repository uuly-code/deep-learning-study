import numpy as np

def softmax(A):
    if A.ndim==2:
        A=A.T
        A=A-np.max(A,axis=0)
        y=np.exp(A)/np.sum(np.exp(A),axis=0)
        return y.T

    A=A-np.max(A)
    return np.exp(A)/np.sum(np.exp(A))


if __name__=="__main__":
    A=np.array([1,2,3])
    print(softmax(A))