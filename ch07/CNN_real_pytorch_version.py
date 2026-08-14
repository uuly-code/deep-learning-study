#输入图片
#--->卷积层：提取特征
#--->ReLU：保留有效特征
#--->池化层：压缩特征图
#--->Affine:综合特征
#--->ReLU
#--->Affine:输出每个类别的分数
#--->Softmax:转成各类别概率

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        #1个输入通道->30个输出通道;卷积核大小5×5
        self.conv1=nn.Conv2d(
            in_channels=1,
            out_channels=30,
            kernel_size=5
        )

        #2×2最大池化，步幅默认为2
        self.pool=nn.MaxPool2d(kernel_size=2)

        #输入：28×28
        #卷积后：24×24
        #池化后：12×12
        #所以展开后：30×12×12
        self.fc1=nn.Linear(30*12*12,100)

        #最终输出10个类别的分数
        self.fc2=nn.Linear(100,10)

    def forward(self,x):
        #卷积->ReLU->池化
        x=self.pool(F.relu(self.conv1(x)))

        #(批量大小，30，12，12)
        x=torch.flatten(x,start_dim=1)

        #Affine->ReLU->Affine
        x=F.relu(self.fc1(x))
        x=self.fc2(x)

        return x

#==测试
model=SimpleCNN()

#假设一次输入4张28×28灰度图
images=torch.randn(4,1,28,28)

scores=model(images)

print("输出形状：",scores.shape)
print(scores)