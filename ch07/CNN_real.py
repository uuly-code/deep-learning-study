import numpy as np
from collections import OrderedDict

# 这些类来自前面章节的实现
from common_layers import Convolution, Relu, Pooling, Affine, SoftmaxWithLoss


class SimpleConvNet:
    def __init__(
        self,
        input_dim=(1, 28, 28),
        conv_param={
            "filter_num": 30,
            "filter_size": 5,
            "pad": 0,
            "stride": 1
        },
        hidden_size=100,
        output_size=10,
        weight_init_std=0.01
    ):
        # 取出卷积层的设置
        filter_num = conv_param["filter_num"]
        filter_size = conv_param["filter_size"]
        filter_pad = conv_param["pad"]
        filter_stride = conv_param["stride"]

        # 卷积后的高、宽
        input_size = input_dim[1]
        conv_output_size = (
            input_size - filter_size + 2 * filter_pad
        ) // filter_stride + 1

        # 池化后展开的数字总数
        pool_output_size = int(
            filter_num
            * (conv_output_size // 2)
            * (conv_output_size // 2)
        )

        # 初始化权重和偏置
        self.params = {}
        self.params["W1"] = weight_init_std * np.random.randn(
            filter_num, input_dim[0], filter_size, filter_size
        )
        self.params["b1"] = np.zeros(filter_num)

        self.params["W2"] = weight_init_std * np.random.randn(
            pool_output_size, hidden_size
        )
        self.params["b2"] = np.zeros(hidden_size)

        self.params["W3"] = weight_init_std * np.random.randn(
            hidden_size, output_size
        )
        self.params["b3"] = np.zeros(output_size)

        # 按顺序搭建 CNN
        self.layers = OrderedDict()
        self.layers["Conv1"] = Convolution(
            self.params["W1"],
            self.params["b1"],
            filter_stride,
            filter_pad
        )
        self.layers["Relu1"] = Relu()
        self.layers["Pool1"] = Pooling(
            pool_h=2,
            pool_w=2,
            stride=2
        )
        self.layers["Affine1"] = Affine(
            self.params["W2"],
            self.params["b2"]
        )
        self.layers["Relu2"] = Relu()
        self.layers["Affine2"] = Affine(
            self.params["W3"],
            self.params["b3"]
        )

        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)

        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        return np.sum(y == t) / x.shape[0]

    def gradient(self, x, t):
        # 正向传播，计算损失
        self.loss(x, t)

        # 从 SoftmaxWithLoss 开始反向传播
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()

        for layer in layers:
            dout = layer.backward(dout)

        # 取出每层算出的梯度
        grads = {}
        grads["W1"] = self.layers["Conv1"].dW
        grads["b1"] = self.layers["Conv1"].db
        grads["W2"] = self.layers["Affine1"].dW
        grads["b2"] = self.layers["Affine1"].db
        grads["W3"] = self.layers["Affine2"].dW
        grads["b3"] = self.layers["Affine2"].db

        return grads