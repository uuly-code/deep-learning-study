import sys ,os
#sys可以操作Python的运行环境，比如修改Python查找文件的路径
#os可以处理文件夹路径
sys.path.append(os.path.join(os.pardir,"ch03")) #os.pardir表示上一级文件夹（这里是study文件夹）
import numpy as np
from common_functions.softmax_function import softmax
import numerical_differ
from loss_functions import cross_entropy_error

class SimpleNet:
    def __init__(self):
        self.W=np.random.randn(2,3)  #表示W是这个网络对象自身拥有的数据
        #给当前网络创建一个权重矩阵W，随机生成一个2（输入神经元的数量）行3（输出神经元的数量）列的矩阵
        #用高斯分布进行初始化W

    def predict(self,x):  #定义预测方法
        return np.dot(x,self.W) #点积

    def loss(self,x,t):
        z=self.predict(x)
        y=softmax(z)
        loss=cross_entropy_error(y,t)

        return loss

net=SimpleNet()
#print(net.W)
x=np.array([0.6,0.9])
p=net.predict(x)
t=np.array([0,0,1])
print(net.loss(x,t))