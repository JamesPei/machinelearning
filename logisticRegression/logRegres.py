# -*- coding:utf-8 -*-
import random

from numpy import mat, shape, ones, exp, array, arange
# import matplotlib.pyplot as plt


# logistic regression梯度上升优化算法
def loadDataSet(path):
    dataMat = []; labelMat = []
    fr = open(path)
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])     # 每个回归系数初始化为1.0
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


def sigmoid(inX):
    return 1.0/(1+exp(-inX))    #exp(x):计算e的x次方


# 梯度上升算法
def gradAscent(dataMatIn, classLabels):
    # 转换为numpy矩阵数据类型
    dataMatrix = mat(dataMatIn)     # Interpret the input as a matrix. Equivalent to matrix(data, copy=False)
    # 为方便矩阵运算，需要将行向量转换为列向量 ??
    labelMat = mat(classLabels).transpose()     # transpose:Returns a view of the array with axes transposed.
    m, n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = ones((n, 1))
    for k in range(maxCycles):
        # 矩阵相乘
        h = sigmoid(dataMatrix * weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights


# 随机梯度上升算法
def stocGradAscent0(dataMatrix, classLabels):
    m, n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights


# 改进的随机梯度上升算法
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m, n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4/(1.0 + j + i) + 0.01          # alpha每次迭代时需要调整.这样可以缓解数据波动或高频波动
            randIndex = int(random.uniform(0, len(dataIndex)))      # 随机选取更新.这样可以减少周期性波动
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights


# def plotBestFit(weights):
#     dataMat, labelMat = loadDataSet('testSet.txt')
#     dataArr = array(dataMat)
#     n = shape(dataArr)[0]
#     xcord1 = []; ycord1 = []
#     xcord2 = []; ycord2 = []
#     for i in range(n):
#         if int(labelMat[i]) == 1:
#             xcord1.append(dataArr[i, 1]); ycord1.append(dataArr[i, 2])
#         else:
#             xcord2.append(dataArr[i, 1]); ycord2.append(dataArr[i, 2])
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
#     ax.scatter(xcord2, ycord2, s=30, c='green')
#     x = arange(-3.0, 3.0, 0.1)
#     y = (-weights[0]-weights[1]*x)/weights[2]           # 最佳拟合直线
#     ax.plot(x,y)
#     plt.xlabel('X1'); plt.ylabel('X2')
#     plt.show()


# 从氙气病症预测病马死亡率
def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5: return 1.0
    else: return 0.0


def colicTest(train_path, test_path):
    frTrain = open(train_path)
    frTest = open(test_path)
    trainingSet = []; trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 500)
    errorCount = 0; numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print("the error rate of this test is: %f" % errorRate)
    return errorRate


def multiTest(train_path, test_path):
    numTests = 10; errorSum=0.0
    for k in range(numTests):
        errorSum += colicTest(train_path, test_path)
    print("after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests)))


if __name__ == '__main__':
    dataMatIn, classLabels = loadDataSet('testSet.txt')
    weights = gradAscent(dataMatIn, classLabels)
    print(weights.getA())   # getA: Return self as an ndarray object. Equivalent to np.asarray(self)