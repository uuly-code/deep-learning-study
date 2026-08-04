import os,sys
sys.path.append(os.pardir)
import numpy as np
from ActivaLayer import *
from gradient_2d import numerical_gradient
from collections import OrderedDict

class TwoLayerNet:
    def __init__(self,input_size,hidden_size,output_size,weight_init_std=0.01):
      #初始化网络  weight_init_std=0.01是初始化权重时使用的缩放值
      #保存参数
        self.params={}
      #创建字典用来保存网络的权重和偏置
        self.params['W1']=weight_init_std*np.random.randn(input_size,hidden_size)
        self.params['b1']=np.zeros(hidden_size)
        self.params['W2']=weight_init_std*np.random.randn(hidden_size,output_size)
        self.params['b2']=np.zeros(output_size)
      #创建一个有顺序的字典，用于按顺序保存网络中的层
        self.layers=OrderedDict()
      #Affine：y=Wx+b
        self.layers['Affine1']=Affine(self.params['W1'],self.params['b1'])
        self.layers['Relu1']=Relu()
        self.layers['Affine2']=Affine(self.params['W2'],self.params['b2'])
      #同时完成使用softmax把得分转换成概率和使用交叉熵误差计算损失
        self.lastlayer=SoftmaxWithLoss()
    
    def predict(self,x):
        for layer in self.layers.values():
            x=layer.forward(x) #上一层的输入成为下一层的输出
        return x

    def loss(self,x,t):  #x:输入图片 t:正确答案标签 y:网络预测出的得分
        y=self.predict(x)   #Affine1-->Relu-->affine2
        return self.lastlayer.forward(y,t)  #Softmax-->交叉熵误差
         #返回一个损失值，预测越接近正确答案，损失值越小

    def accuracy(self,x,t):  #计算正确率
        y=self.predict(x)
        y=np.argmax(y,axis=1)
        if t.ndim !=1:
            t=np.argmax(t,axis=1)
        accuracy=np.sum(y==t)/float(x.shape[0])  #正确率
        #y==t:逐个判断预测是否正确 np.sum统计数量
        return accuracy

     #数值微分求梯度 #数值微分比较慢，主要用于检查后面的误差反向传播是否正确
    def numerical_gradient(self,x,t):
        loss_W=lambda W:self.loss(x,t)
       #定义一个临时函数，表示根据当前权重计算损失
        grads={}
        grads['W1']=numerical_gradient(loss_W,self.params['W1'])
        grads['b1']=numerical_gradient(loss_W,self.params['b1'])
        grads['W2']=numerical_gradient(loss_W,self.params['W2'])
        grads['b2']=numerical_gradient(loss_W,self.params['b2'])

        return grads
      #反向传播求梯度
    def gradient(self,x,t):
        self.loss(x,t) #先进行向前传播

       #从最后一层开始反向传播 #Affine2-->Relu-->Affine1
        dout=1
        dout=self.lastlayer.backward(dout)

        layers=list(self.layers.values())
        layers.reverse() #翻转
        for layer in layers:
            dout=layer.backward(dout)
        
        grads={}
        grads['W1']=self.layers['Affine1'].dW
        grads['b1']=self.layers['Affine1'].db
        grads['W2']=self.layers['Affine2'].dW
        grads['b2']=self.layers['Affine2'].db

        return grads
