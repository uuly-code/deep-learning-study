import numpy as np
import matplotlib.pyplot as plt

class Student:
    def __init__(self,name,scores):
        self.name=name
        self.scores=np.array(scores)
    def average(self):
        aver=np.mean(self.scores)
        return aver
    def introduce(self):
        print(f"{self.name}的平均分为{round(self.average(),2)}")

s1=Student("小玉",[78,85,92])
s2=Student("小明",[60,72,68])
s3=Student("小林",[45,58,50])

students=[s1,s2,s3]

for i in students:
    i.introduce()
    if i.average()>=60.0:
        print("及格")
    else:
        print("不及格")

x=np.array([1,2,3])
y1=s1.scores 
y2=s2.scores 
y3=s3.scores
plt.plot(x,y1,label="xiao yu")
plt.plot(x,y2,label="xiao ming")
plt.plot(x,y3,label="xiao lin")
plt.xlabel("test_number")
plt.xticks([1,2,3])   #让x轴只显示整数
plt.ylabel("scores")
plt.legend()
plt.show()