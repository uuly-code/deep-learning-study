import numpy as np

x=np.array([
    [1,2,3,4],
    [5,7,7,8],
    [9,10,11,12],
    [13,14,15,16]
])

W=np.array([
    [
        [1,0],
        [0,1]
    ],
    [
        [0,1],
        [1,0]
    ]
])

#im2col
windows=[]

for y in range(3):
    for x_pos in range(3):
        window=x[y:y+2,x_pos:x_pos+2]

        #将2×2窗口拉成一行
        windows.append(window.flatten())

#所有窗口组成一个大矩阵
col=np.array(windows)

print("im2col后的矩阵:")
print(col)

#将卷积核拉成一列
W_col=W.reshape(2,-1).T

print("\n卷积核展开后:")
print(W_col)

#每个窗口与卷积核做点积
result=np.dot(col,W_col)

print("\n矩阵乘法结果:")
print(result)

#恢复为输出特征值

out=result.reshape(3,3,2).transpose(2,0,1)

print("\n最终输出:")
print(out)

#卷积层的反向传播
def backward(self,dout):
    FN,C,FH,FW=self.W.shape

    #让dout也变成“每个窗口一行”
    dout=dout.transpose(0,2,3,1).reshape(-1,FN)

    #偏置的梯度
    self.db=np.sum(dout,axis=0)

    #卷积核的梯度：告诉W中每个数字该怎样调整
    self.dW=np.dot(self.col.T,,dout)
    self.dW=self.dW.transpose(1,0).reshape(FN,C,FH,FW)

    #把误差传回输入
    dcol=np.dot(dout,self.col_W.T)
    dx=col2im(dcol,self.x.shape,FH,FW,self.stride,self.pad)
                                        #stride:步幅 pad:填充
    return dx