import numpy as np

A=np.array([1,2])
B=np.array([[1,3,5],
           [2,4,6]])
C=np.array([[1,3],[2,5],[3,6]])
D=np.array([[1,2],[3,4]])

M=np.dot(A,B)
print(M)
print("\n")
N=np.dot(M,C)
print(N)
print("\n")
P=np.dot(N,D)
print(P)
