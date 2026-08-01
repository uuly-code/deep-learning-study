import numpy as np
import matplotlib.pylab as plt

def numerical_diff(f,x):
    h=1e-4  #0.0001
    return(f(x+h)-f(x-h))/(2*h)

def function_1(x):
    return 0.01*x**2+0.1*x

# x=np.arange(0.0,20.0,0.1) #以0.1为单位，从0-20的数组x
# y1=function_1(x)
# y2=numerical_diff(function_1,5)*(x-5)+function_1(5)
# plt.xlabel("x")
# plt.ylabel("f(x)")
# plt.plot(x,y1,label='f(x)')
# plt.plot(x,y2,label='x=5qiexian')
# plt.legend()
# plt.show()

#函数在x=5处的导数
#print(numerical_diff(function_1,5))


def function_2(x):
    return np.sum(x**2,axis=0)

# x0=np.arange(-4,4.1,0.1)
# x1=np.arange(-4,4.1,0.1)

# #生成坐标网络   把x1 x0所有可能组合整理成网格
# x0,x1=np.meshgrid(x0,x1)

# #将x0和x1组合起来传入函数
# X=np.array([x0,x1])
# Z=function_2(X)

# #创建三维坐标系
# fig=plt.figure()  #创建画布
# ax=fig.add_subplot(111,projection="3d")   #1行1列第1张图

# #绘制网格图
# #ax.plot_wireframe(x0,x1,Z)
# ax.plot_surface(x0,x1,Z)  #连续曲面，非网格图

# ax.set_xlabel("x0")
# ax.set_xlabel("x1")
# ax.set_zlabel("f(x0,x1)")

# plt.show()


#求x0的偏导数，若在（3，4）点处，则把x1=4固定
def function_tmp1(x0):
    return x0**2+4.0**2

#print(numerical_diff(function_tmp1,3.0))

#求x1的偏导数，若在（3，4）点处，则把x0=3固定
def function_tmp2(x1):
    return 3.0**2+x1**2
#print(numerical_diff(function_tmp2,4.0))



#梯度
def numerical_gradient(f,x):
    h=1e-4
    grad=np.zeros_like(x)  #生成和x形状一样的数组

    for idx in np.ndindex(x.shape):
        tmp_val=x[idx]
        #f(x+h)的计算
        x[idx]=tmp_val+h
        fxh1=f(x)

        #f(x-h)的计算
        x[idx]=tmp_val-h
        fxh2=f(x)

        grad[idx]=(fxh1-fxh2)/(2*h)
        x[idx]=tmp_val  #还原值
    return grad


#梯度下降法
def gradient_descent(f,init_x,lr=0.01,step_num=100):
    x=init_x

    for i in range(step_num):
        grad=numerical_gradient(f,x)
        x-=lr*grad
    return x

init_x=np.array([-3.0,4.0])

if __name__=="__main__":
    print(gradient_descent(function_2,init_x=init_x,lr=0.1,step_num=100))