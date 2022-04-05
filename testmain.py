import torch
import numpy as np
from torchvision import transforms
import math

def transto3D(twoDimensionList):
    num=twoDimensionList.shape[0]
    n = len(twoDimensionList[0])
    edge = int(math.sqrt(n) + 1)
    threeDList = torch.zeros(num,edge, edge)
    for numcount in range(num):
        i = 0
        for j in range(edge):
            if i >= n: break
            for k in range(edge):
                if i>=n:break
                threeDList[numcount,j, k] = twoDimensionList[numcount][i]
                i+=1
    result = threeDList.clone().detach()
    return result

def f(min,max):
    s=min
    for i in range(min+1,max+1):
        s=s*i
    return s
if __name__ == '__main__':
    # x=torch.ones(3,27)
    # y=transto3D(x)
    # print(y)
    # y=y[:,np.newaxis, :,:]
    # print(x)
    # print(y)
    s=0
    while(True):
        n=int(input('请输入：'))
        for i in range(1,n+1):
            s=s+f(n-i,n)
        print(s)
        s=0