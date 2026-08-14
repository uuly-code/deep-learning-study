
import numpy as np
def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def cross_entropy_error(y, t):
    if t.ndim == 1:
        t = np.eye(y.shape[1])[t]
    delta = 1e-7
    return -np.sum(t * np.log(y + delta)) / y.shape[0]


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """(N, C, H, W) 的图片批量 -> 每个局部窗口一行的二维矩阵。"""
    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    img = np.pad(input_data, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    return col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """im2col 的反向操作：把局部窗口的梯度加回输入位置。"""
    N, C, H, W = input_shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w)
    col = col.transpose(0, 3, 4, 5, 1, 2)
    img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]


class Relu:
    def forward(self, x):
        self.mask = x <= 0
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        return dout


class Affine:
    def __init__(self, W, b):
        self.W, self.b = W, b

    def forward(self, x):
        self.original_x_shape = x.shape
        self.x = x.reshape(x.shape[0], -1)
        return np.dot(self.x, self.W) + self.b

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx.reshape(*self.original_x_shape)


class SoftmaxWithLoss:
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        return cross_entropy_error(self.y, self.t)

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.ndim == 1:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
        else:
            dx = self.y - self.t
        return dx * dout / batch_size


class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W, self.b = W, b
        self.stride, self.pad = stride, pad

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, _, H, W = x.shape
        out_h = (H + 2 * self.pad - FH) // self.stride + 1
        out_w = (W + 2 * self.pad - FW) // self.stride + 1

        self.x = x
        self.col = im2col(x, FH, FW, self.stride, self.pad)
        self.col_W = self.W.reshape(FN, -1).T
        out = np.dot(self.col, self.col_W) + self.b
        return out.reshape(N, out_h, out_w, FN).transpose(0, 3, 1, 2)

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)
        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout).T.reshape(FN, C, FH, FW)
        dcol = np.dot(dout, self.col_W.T)
        return col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)


class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h, self.pool_w = pool_h, pool_w
        self.stride, self.pad = stride, pad

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = (H - self.pool_h) // self.stride + 1
        out_w = (W - self.pool_w) // self.stride + 1

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h * self.pool_w)
        self.arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        self.x = x
        return out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)
        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max] = dout.flatten()
        dcol = dmax.reshape(-1, pool_size)
        return col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
