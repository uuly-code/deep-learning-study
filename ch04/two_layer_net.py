import sys,os
sys.path.append(os.path.join(os.pardir,"ch03"))
from common_functions.sigmoid import sigmoid 
from common_functions.softmax_function import softmax 
from numerical_differ import numerical_gradient
import numpy as np
from loss_functions import cross_entropy_error
class TwoLayerNet:

    def __init__(self,input_size,hidden_size,output_size,weight_init_std=0.01):
        #初始化权重
        self.params={} #创建参数字典
        #'W1':第一层权重
        #'b1':第一层偏置
        #'W2':第二层权重
        #'b2':第二层偏置
        self.params['W1']=weight_init_std*\
                          np.random.randn(input_size,hidden_size)
        self.params['b1']=np.zeros(hidden_size)
        self.params['W2']=weight_init_std*\
                          np.random.randn(hidden_size,output_size)
        self.params['b2']=np.zeros(output_size)

    def predict(self,x):
        W1,W2=self.params['W1'],self.params['W2']
        b1,b2=self.params['b1'],self.params['b2']

        a1=np.dot(x,W1)+b1
        z1=sigmoid(a1)
        a2=np.dot(z1,W2)+b2
        y=softmax(a2)

        return y

    def loss(self,x,t):
        y=self.predict(x)

        return cross_entropy_error(y,t)

    def accuracy(self,x,t):
        y=self.predict(x)
        y=np.argmax(y,axis=1)
        t=np.argmax(t,axis=1)

        accuracy=np.sum(y==t)/float(x.shape[0])
        return accuracy

    def numerical_gradient(self,x,t):
        loss_W=lambda W:self.loss(x,t)
        #等价于:
        #def loss_W(W):
        #    return self.loss(x,t)
        grads={}
        grads['W1']=numerical_gradient(loss_W,self.params['W1'])
        grads['b1']=numerical_gradient(loss_W,self.params['b1'])
        grads['W2']=numerical_gradient(loss_W,self.params['W2'])
        grads['b2']=numerical_gradient(loss_W,self.params['b2'])

        return grads