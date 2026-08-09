class SGD:
    def __init__(self,lr=0.1):
        self.lr=lr    #学习率为0.1

    def update(self,params,grads):
        for key in params.keys():  #逐个处理每一个参数
            params[key]-=self.lr*grads[key]

#伪代码
network=TwoLayerNet(...)
optimizer=SGD()

for i in range(10000):
    ...
    x_batch,t_batch=get_mini_batch(...)  #mini-batch
    grads=network.gradient(x_batch,t_batch)
    params=network.params
    optimizer.update(params,grad)