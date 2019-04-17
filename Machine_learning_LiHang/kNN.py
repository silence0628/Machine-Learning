'''
Created on Sep 16, 2010
kNN: k Nearest Neighbors

Input:      inX: vector to compare to existing dataset (1xN)
            dataSet: size m data set of known vectors (NxM)
            labels: data set labels (1xM vector)
            k: number of neighbors to use for comparison (should be an odd number)
            
Output:     the most popular class label

@author: pbharrin
'''
#%%
import numpy as np
import operator
from os import listdir

#%%
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]  #  4
    diffMat = np.tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    #    array([[ 1.  ,  1.21],
    #       [ 1.  ,  1.  ],
    #       [ 0.  ,  0.  ],
    #       [ 0.  ,  0.01]])
    sqDistances = sqDiffMat.sum(axis=1)  #  array([ 2.21,  2.  ,  0.  ,  0.01])
    distances = sqDistances**0.5  # array([ 1.48660687,  1.41421356,  0.        ,  0.1       ])
    sortedDistIndicies = distances.argsort()      #  array([2, 3, 1, 0], dtype=int64)
    classCount={}          
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

#%%
def createDataSet():
    group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels
group, labels = createDataSet()
pred = classify0([0.5,0.5],group,labels,3)
#%%
#加载数据 读取txt文件，转换成 数据矩阵 ，和标签矩阵 
def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()  #注意需要加s
    numberOfLines = len(arrayOLines)            #get the number of lines in the file  1000
    returnMat = np.zeros((numberOfLines,3))     #prepare matrix to return
    classLabelVector = []                       #prepare labels return 
    index = 0
    for line in arrayOLines:
        line = line.strip()
#        print(line)
        listFormLine = line.split('\t')
#        print(listFormLine)
        for x in range(0,3):
            returnMat[index,x] = float(listFormLine[x])
#        classLabelVector.append(int(listFormLine[-1])) # -1 为最后一个元素  适用于  datingTestSet2.txt
        classLabelVector.append(int(listFormLine[-1])) # -1 为最后一个元素  适用于  datingTestSet2.txt
        index += 1
    return returnMat,classLabelVector
dataTestDir = 'E:\HHH乱\机器学习实战\机器学习实战源代码\machinelearninginaction\Ch02\datingTestSet2.txt'
datingDataMat, datingLabels = file2matrix(dataTestDir)
#%%
#"""绘图 ， 数据可视化"""
#import matplotlib.pyplot as plt
#from pylab import mpl
#"""  1  """
#mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
#mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
#a = datingDataMat[:,0]  #  每年获取的飞行常客里程数
#b = datingDataMat[:,1]  #  玩视频游戏所耗时间百分比 
#c = datingDataMat[:,2]  #  每周所消费的冰淇淋公升数 
#fig = plt.figure(figsize=(10,10))
#ax = fig.add_subplot(111)
#
#ax.scatter(datingDataMat[:,1],datingDataMat[:,2],15.0*np.array(datingLabels),15.0*np.array(datingLabels))  
#plt.xlabel(u'玩视频游戏所耗时间百分比', fontsize = 16)
#plt.ylabel(u'每周所消费的冰淇淋公升数', fontsize = 16)
#plt.title(u'约会数据可视化图 1 ',fontsize = 20)
#plt.show()
#"""没有样本类别标签的约会数据散点图。难以辨识图中的点究竟属于哪个样本分类"""
##%%
#import matplotlib.pyplot as plt
#from pylab import mpl
#"""  2  """
#mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
#mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
#a = datingDataMat[:,0]  #  每年获取的飞行常客里程数
#b = datingDataMat[:,1]  #  玩视频游戏所耗时间百分比 
#c = datingDataMat[:,2]  #  每周所消费的冰淇淋公升数 
#fig = plt.figure(figsize=(10,10))
#ax = fig.add_subplot(111)
#
#ax.scatter(datingDataMat[:,0],datingDataMat[:,1],15.0*np.array(datingLabels),15.0*np.array(datingLabels))  
#plt.xlabel(u'每年获取的飞行常客里程数', fontsize = 16)
#plt.ylabel(u'玩视频游戏所耗时间百分比', fontsize = 16)
#plt.title(u'约会数据可视化图 2 ',fontsize = 20)
#plt.show()
#%%    
"""归一化特征值，里程值、所耗时间、消耗冰淇淋数三种特征值的取值范围不一致"""
datingDataMat, datingLabels = file2matrix(dataTestDir)
#%%
def autoNorm(dataSet):
    """dataSet.shape=(1000,3),按照 列 找到 每一列即每一种特征值的最大和最小值，即取值范围"""
    minVals = dataSet.min(0)   #  [ 0.      ,  0.      ,  0.001156]
    maxVals = dataSet.max(0)   #  [  9.12730000e+04,   2.09193490e+01,   1.69551700e+00]
    ranges = maxVals - minVals #  [  9.12730000e+04,   2.09193490e+01,   1.69436100e+00]
    normDataSet = np.zeros(np.shape(dataSet))  #  生成 等大小 空矩阵 
    m = dataSet.shape[0]  #  1000
    normDataSet = dataSet - np.tile(minVals, (m,1))
    normDataSet = normDataSet/np.tile(ranges, (m,1))   #element wise divide
    return normDataSet, ranges, minVals

#normDataSet, ranges, minVals =  autoNorm(datingDataMat)

#%%   
"""分类器针对约会网站的测试代码"""
dataTestDir = 'E:\HHH乱\机器学习实战\机器学习实战源代码\machinelearninginaction\Ch02\datingTestSet2.txt'

def datingClassTest(dirs):
    hoRatio = 0.50      # hold out 10%
    datingDataMat,datingLabels = file2matrix(dirs)       #load data setfrom file
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]     #   1000 
    numTestVecs = int(m*hoRatio)      #  500
    errorCount = 0.0
    """整个数据集，前500个做test，后500个做train"""
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],8)
#        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
    print("the total error rate is: %f" % (errorCount/float(numTestVecs))) 
    print(errorCount)

datingClassTest(dataTestDir)
#%%
"""k=1,error_rate=0.080000  
   k=2,error_rate=0.080000
   k=2,error_rate=0.066000
   k=2,error_rate=0.062000"""

#for i in range(1,5):
#    datingClassTest(dataTestDir,i)
#    print('*'*40)
























#%%
"""手写数字识别，knn验证算法"""
def img2vector(filename):
    returnVect = np.zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j]) 
    return returnVect
#%%
import os
train_dir = r'E:\HHH乱\机器学习实战\机器学习实战源代码\machinelearninginaction\Ch02\digits\\trainingDigits\\'
test_dir = r'E:\HHH乱\机器学习实战\机器学习实战源代码\\machinelearninginaction\\Ch02\\digits\\testDigits\\'
def handwritingClassTest(train_dir, testDigits):
    hwLabels = []
    trainingFileList = os.listdir(train_dir)  #load the training set
    m = len(trainingFileList)    #  1934
    trainingMat = np.zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     # 0_0.txt
        classNumStr = int(fileStr.split('_')[0]) #  0
        hwLabels.append(classNumStr)
#        print(train_dir+'\\%s' % fileNameStr)
        trainingMat[i,:] = img2vector(train_dir+'\\%s' % fileNameStr)
    
    testFileList = listdir(testDigits)        #iterate through the test set
    print('训练数据和测试数据准备完毕，准备训练并测试')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector(testDigits+'\\%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
#        print("KNN 分类器返回值是: %d, 真实值是: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print("\n错误总数目为 : %d" % errorCount) 
    print ("\n最终错误率为 : %f" % (100*errorCount/float(mTest)) + '%')

handwritingClassTest(train_dir, test_dir)
"""the total number of errors is: 10
   the total error rate is: 0.010571 """





