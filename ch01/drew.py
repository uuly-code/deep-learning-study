import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread 

# x=np.arange(0,5,0.1)   #以0.1为单位，生成0-5的单位
# y1=x
# y2=x*x

# plxt.plot(x,y1,label="y=x")
# plt.plot(x,y2,linestyle="--",label="y=x^2")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.title('Two Functions')
# plt.legend()
# plt.show()
img=imread('屏幕截图 2026-07-16 140537.png')
plt.imshow(img)
plt.show()