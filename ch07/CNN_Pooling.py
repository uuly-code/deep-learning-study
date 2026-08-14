import numpy as np

x=np.array([
    [2,0,0,6],
    [1,0,1,1],
    [2,0,2,6],
    [0,8,1,4]
])

pool_h=2    #池化窗口高度
pool_w=2    #池化窗口宽度
stride=2    #每次移动2格

windows=[]

for y in range(0,4,stride):
    for x_pos in range(0,4,stride):
        window=x[y:y+pool_h,x_pos:x_pos+pool_w]

        windows.append(window.flatten())

col=np.array(windows)

print("im2col后:")
print(col)

#每一行取最大值
result=np.max(col,axis=1)

print("\n每个窗口的最大值:")
print(result)

#重新排成输出特征图
out=result.reshape(2,2)

print("\n最大池化输出:")
print(out)