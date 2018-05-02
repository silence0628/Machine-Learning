# -*- coding: utf-8 -*-
"""
Created on Tue May  1 20:26:05 2018

@author: huhaohang

content: 主要是针对三种回归评价机制对回归性能进行研究
"""
from sklearn.datasets import load_boston # 波士顿房价数据集
boston = load_boston()

from sklearn.cross_validation import train_test_split  #  切割数据
import numpy as np
X = boston.data
y = boston.target
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=33,test_size=0.25)
print("The max target value is ",np.max(boston.target))
print("The min target value is ",np.min(boston.target))
print("The mean target value is ",np.mean(boston.target))
#%%
from sklearn.preprocessing import StandardScaler # 数据标准化模块
ss_X = StandardScaler()  #  初始化
ss_y = StandardScaler()  #  初始化
X_train = ss_X.fit_transform(X_train)
X_test = ss_X.transform(X_test)

# 将数据格式重构  （xxx,）==>> (xxx,1)
y_train = y_train.reshape(len(y_train),1)
y_test = y_test.reshape(len(y_test),1)
y_train = ss_y.fit_transform(y_train)
y_test = ss_y.transform(y_test)

#%%
# 线性回归器
from sklearn.linear_model import LinearRegression
lr = LinearRegression()# 使用默认配置初始化回归器
lr.fit(X_train,y_train)
lr_y_predict = lr.predict(X_test)# 预测

# SGD线性回归器
from sklearn.linear_model import SGDRegressor
sgdr =SGDRegressor()
sgdr.fit(X_train,y_train)
sgdr_y_predict = sgdr.predict(X_test)

#%%
"""使用3种回归评价机制以及2种调用R-squared评价模块的方法，对2种回归模型的回归性能进行评价"""
#使用LinearRegression模型自带的评估模块，并输入结果
"""从sklearn.metrics依次导入 r2_score, mean_squared_error均方误差, mean_absolute_error平均绝对误差 评价 线性回归 性能"""

from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error

Linear_result = lr.score(X_test,y_test)
print('The value of default measurement of LinearRegression is ',Linear_result)
print('The value of R-squared of LinearRegression is ',r2_score(y_test,lr_y_predict))
print('The mean squared error of LinearRegression is',mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(lr_y_predict)))
print('The mean absolute error of LinearRegression is',mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(lr_y_predict)))

print('*'*80)
SGDRegressor_result = lr.score(X_test,y_test)
print('The value of default measurement of SGDRegression is ',SGDRegressor_result)
print('The value of R-squared of SGDRegression is ',r2_score(y_test,sgdr_y_predict))
print('The mean squared error of SGDRegression is',mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(sgdr_y_predict)))
print('The mean absolute error of SGDRegression is',mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(sgdr_y_predict)))




















