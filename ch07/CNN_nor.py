import numpy as np
 
x=np.array([
    [1,2,3,4],
    [5,6,7,8],
    [9,10,11,12],
    [13,14,15,16]
])

 #卷积核
W=np.array([
    [1,0],
    [0,1]
])

 #输出大小
out=np.zeros((3,3))

for y in range(3):
    for x_pos in range(3):
        #取出卷积核当前盖住的局部窗口
        window=x[y:y+2,x_pos:x_pos+2]

        #元素对应相乘，再全部求和
        out[y,x_pos]=np.sum(window*W)


        print(f"位置({y},{x_pos})")
        print("窗口：")
        print(window)
        print("结果：",out[y,x_pos])
        print()

print("最终输出：")
print(out)