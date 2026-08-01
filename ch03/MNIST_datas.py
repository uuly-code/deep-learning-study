# from dataset.mnist import load_mnist
# import matplotlib.pyplot as plt

# (x_train,t_train),(x_test,t_test)=load_mnist(
#     normalize=True,     #像素值保持在0~255  若为True则保持在0~1
#     flatten=True
# )

# print(x_train.shape)
# print(t_train.shape)
# print(x_test.shape)
# print(t_test.shape)
# i=15
# print(f"第{i}张图片的答案：",t_train[i])

# img=x_train[i].reshape(28,28)
# plt.imshow(img,cmap='gray')
# plt.show()



import sys ,os
sys.path.append(os.pardir)   #为了导入父目录中的文件而进行的设定
from dataset.mnist import load_mnist

#第一次调试可能会花费几分钟
(x_train,t_train),(x_test,t_test)=load_mnist(flatten=False,normalize=False,one_hot_label=False)

#输出各个数据的形状
print(x_train.shape)  
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)
print(t_train[5])