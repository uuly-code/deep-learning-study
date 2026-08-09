
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    """Sigmoid激活函数"""
    return 1/(1+np.exp(-x))

def relu(x):
    """Relu激活函数"""
    return np.maximum(0,x)

#固定随机数，保证每次运行结果相同
np.random.seed(0)

#生成1000条数据,每条数据有100个特征
x=np.random.randn(1000,100)

#每层神经元数量
node_num=100
#隐藏层数量
hidden_layer_size=5
#保存每一层的激活值
activations={}

#选择权重初始化的方法
"""
   std1:标准差为1
   std001:标准差为001
   xavier:Xavier初始化
   he:He初始化
"""
weight_init="he"
#选择激活函数
#"sigmoid"或"relu"
activation_function="relu"

for i in range(hidden_layer_size):
    if i !=0:
        x=activations[i-1] #第一层使用原始输入

    #根据选择生成权重
    if weight_init == "std1":#生成服从正态分布的随机数
        w=np.random.randn(node_num,node_num)

    elif weight_init=="std001":
        w=np.random.randn(node_num,node_num)*0.01

    elif weight_init=="xavier":#使输入信号经过每一层后数值大小尽量保持稳定，适合用Sigmoid函数
        w=(np.random.randn(node_num,node_num)*np.sqrt(1.0/node_num))

    elif weight_init=="he":#适合用Relu函数
        w=(np.random.randn(node_num,node_num)*np.sqrt(2.0/node_num))
    else:
        raise ValueError("未知的权重初始化办法")

    z=np.dot(x,w)

    # 激活函数
    if activation_function == "sigmoid":
        a = sigmoid(z)

    elif activation_function == "relu":
        a = relu(z)

    else:
        raise ValueError(
            f"未知的激活函数：{activation_function}"
        )


    #保存当前层的激活值
    activations[i]=a

#绘制每一层激活值的直方图
plt.figure(figsize=(15,4))

for i,a in activations.items():
    plt.subplot(1,hidden_layer_size,i+1)
    plt.title(str(i+1)+"-layer")

    #除第一张图外隐藏纵坐标标签
    if i != 0:
        plt.yticks([])
    
    plt.hist(a.flatten(),bins=30,range=(0,1))

plt.tight_layout()
plt.show()