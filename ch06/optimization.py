#新速度=旧速度+当前梯度保留的速度
#带着之前积累的速度继续走，从而加快稳定方向上的移动并减小来回振荡
import numpy as np


class Momentum:
    def__init__(self,lr=0.01,momentum=0.9):
        self.lr=lr
        self.momentum=momentum
        self.v=None

    def update(self,params,grads):

        #第一次更新，为每个参数建立一个相同形状的速度，并全部初始化为0
        if self.v is None:
            self.v={}
            for key,val in params.items():
                self.v[key]=np.zeros_like(val)
        #真正的更新代码
        for key in params.keys():
            self.v[key]=self.momentum*self.v[key]-self.lr*grads[key]
            params[key]+=self.v[key]



#记录每个参数的梯度平方和，过去梯度越大的参数，今后的更新幅度就越小
class AdaGrad:
    def __init__(self,lr=0.1):
        self.lr=lr
        self.h=None

    def update(self,params,grads):
        if self.h is None:
        self.h={}
        for key,val in params.items():
            self.h[key]=np.zeros_like(val)
    
    for key in params.key():
        self.h[key]+=grads[key]*grads[key]
        params[key]-=self.lr*grads[key]/(np.sqrt(self.h[key])+1e-7)
            
        
