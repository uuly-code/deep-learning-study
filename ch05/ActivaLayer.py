import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),"..","ch03"))
sys.path.append(os.path.join(os.path.dirname(__file__),"..","ch04"))
from loss_functions import cross_entropy_error
from common_functions.softmax_function import softmax
class Relu:
    def __init__(self):
        self.mask=None  #mask用来记录哪些位置小于或等于0

    def forward(self,x):
        self.mask=(x<=0) #小于等于0的话就是True，否则为False
        out=x.copy()   #保留一份x原本的值
        out[self.mask]=0 #把mask中True的位置设为0

        return out

    def backward(self,dout):
        dx=dout.copy()
        dx[self.mask]=0

        return dx

class Sigmoid:
    def __init__(self):
        self.out=None

    def forward(self,x):
        out=1/(1+np.exp(-x))
        self.out=out
        return out

    def backward(self,dout):
        dx=dout*(1.0-self.out)*self.out

        return dx

class Affine:
    def __init__(self,W,b):
        self.W=W
        self.b=b
        self.x=None
        self.dW=None
        self.db=None

    def forward(self,x):
        self.x=x
        out=np.dot(x,self.W)+self.b
        return out

    def backward(self,dout):
        dx=np.dot(dout,self.W.T)  #.T表示转置，使矩阵形状能够相乘
        self.dW=np.dot(self.x.T,dout)
        self.db=np.sum(dout,axis=0)
        return dx        


class SoftmaxWithLoss:
    def __init__(self):
        self.loss=None  #损失
        self.y=None    #softmax的输出
        self.t=None    #监督数据

    def forward(self,x,t):
        self.t=t
        self.y=softmax(x)
        self.loss=cross_entropy_error(self.y,self.t)

        return self.loss

    def backward(self,dout=1):
        batch_size=self.t.shape[0]
        dx=(self.y-self.t)/batch_size

        return dx


#x=np.array([[1,-0.5],[-2.0,3]])
#mask=(x<=0)
#print(mask)