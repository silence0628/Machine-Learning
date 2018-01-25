#coding:utf-8
"""
Created on Thu Jan 25 20:19:04 2018

@author: huhaohang
"""
import numpy as np
print('--------------------------------------------------------------------------')
input_ = np.array([3,3,4,3,1,1]).reshape((3,2))   #  分类数据
y_ = np.array([1,1,-1])   # 分类标签
length = len(input_)

"""sign  点乘函数"""
def sign(w,b,x):  
    y = np.dot(x,w)+b  
    return int(y)  
"""初始化，全部取0 ，w_=array([[0.],[0.]])  b_=0 ，以及学习率取 n=1 """

w_ = np.zeros((input_.shape[1],1)) 
b_ = 0
n =1   
print('--------------------------------------------------------------------------')
flag = True
k=0
while flag:
    count = 0
    k+=1
    for i in range(length):
        
        result = sign(w_,b_,input_[i,:])
#        print(y_)
        if result*y_[i] <=0:
            
            tmp = y_[i]*n*input_[i,:]  
            tmp = tmp.reshape(w_.shape)  
            w_ = tmp +w_  
            b_ = b_ + y_[i]
            print(w_,b_)
            print('xxxxxxxxxxxxxxxxxxx')
            
            count +=1
    if count == 0:
        print('结束')
        flag = False
print('--------------------------------------------------------------------------')
"""k表示总的迭代次数，这里和课本上不同是因为，在第五次迭代时，误分类点选取的不同导致的"""
print(k)  
"""最终生成的 权重矩阵 ，偏置 b_"""
print(w_,b_)    
#%%