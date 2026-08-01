import numpy as np
def AND (x1,x2):
    x=np.array([x1,x2])
    w=np.array([1,1])
    b=-1
    tmp=b+np.sum(w*x)
    if tmp<=0:
        return 0
    else: 
        return 1

print(AND(0,0))
print(AND(1,0))
print(AND(0,1))
print(AND(1,1))
print("\n")

def NAND (x1,x2):
    x=np.array([x1,x2])
    w=np.array([-1,-1])
    b=1.1
    tmp=b+np.sum(w*x)
    if tmp<=0:
        return 0
    else: 
        return 1

print(NAND(0,0))
print(NAND(1,0))
print(NAND(0,1))
print(NAND(1,1))
print("\n")

def OR (x1,x2):
    x=np.array([x1,x2])
    w=np.array([1,1])
    b=0
    tmp=b+np.sum(w*x)
    if tmp<=0:
        return 0
    else: 
        return 1

print(OR(0,0))
print(OR(1,0))
print(OR(0,1))
print(OR(1,1))
print("\n")

#多层感知机
def XOR (x1,x2):
    s1=NAND(x1,x2)
    s2=OR(x1,x2)
    return AND(s1,s2)

print(XOR(0,0))
print(XOR(1,0))
print(XOR(0,1))
print(XOR(1,1))
