import os,sys
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from TwoLayerNet import TwoLayerNet

(x_train,t_train),(x_test,t_test)=load_mnist(normalize=True,one_hot_label=True)
network=TwoLayerNet(input_size=784,hidden_size=50,output_size=10)
x_batch=x_train[:3]
t_batch=t_train[:3]

#用数值微分和误差反向传播分别计算同一组梯度，比较两者是否接近，用来检验反向传播代码是否正确
grad_numerical=network.numerical_gradient(x_batch,t_batch)
grad_backprop=network.gradient(x_batch,t_batch)

for key in grad_numerical.keys():
    diff=np.average(np.abs(grad_backprop[key]-grad_numerical[key]))
    print(key+":"+str(diff))