# -*- coding:utf-8 -*-
import random

from numpy import mat, shape, zeros, multiply

"""
SVM的伪代码表示：
创建一个alpha向量并将其初始化为0向量
当迭代次数小于最大迭代次数时（外循环）
    对数据集中的每个数据向量（内循环）：
        如果该数据向量可以被优化：
            随机选择另外一个数据向量
            同时优化这两个向量
            如果两个向量都不能被优化，退出内循环
    如果所有向量都没有被优化，增加迭代数目，继续下一次循环
"""


def loadDataSet(fileName):
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat


def selectJrand(i, m):
    '''
    :param i: 第一个alpha的下标
    :param m: 所有alpha的数目
    :return:
    '''
    j = i
    while(j==i):
        j = int(random.uniform(0, m))
    return j


def clipAlpha(aj, H, L):
    '''调整大于H或小于L的alpha值'''
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    """
    :param dataMatIn: 数据集
    :param classLabels: 类别标签
    :param C: 常数C
    :param toler: 容错率
    :param maxIter: 退出前最大循环次数
    :return:
    """
    dataMatrix = mat(dataMatIn)                 # Interpret the input as a matrix.
    labelMat = mat(classLabels).transpose()
    b = 0; m, n = shape(dataMatrix)
    alphas = mat(zeros((m, 1)))
    iter = 0
    while (iter < maxIter):
        alphaPairsChanged = 0
        for i in range(m):
            fXi = float(multiply(alphas, labelMat).T*(dataMatrix*dataMatrix[i, :].T)) + b   # Multiply arguments element-wise.
            Ei = fXi - float(labelMat[i])
            # 如果alpha可以更改进入优化过程
            if((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i, m)       # 随机选择第二个alpha
                fXj = float(multiply(alphas, labelMat).T*(dataMatrix*dataMatrix[j, :].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                # 保证alpha在0与C之间
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H: print("L==H"); continue
                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - dataMatrix[i, :] * dataMatrix[i, :].T -\
                      dataMatrix[j, :] * dataMatrix[j, :].T         # alpha[j]的最优修改量
                if eta >= 0: print("eta >= 0"); continue
                alphas[j] -= labelMat[j]*(Ei - Ej)/eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                if(abs(alphas[j] - alphaJold) < 0.00001): print("j not moving enough"); continue
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])      # 对i进行修改，改量与j相同，但方向相反
                b1 = b - Ei - labelMat[i]*(alphas[i] - alphaIold)*dataMatrix[i, :]*dataMatrix[i, :].T -\
                     labelMat[j]*(alphas[j] - alphaJold)*dataMatrix[i, :]*dataMatrix[j, :].T
                b2 = b - Ej - labelMat[i]*(alphas[i] - alphaIold)*dataMatrix[i, :]*dataMatrix[j, :].T -\
                     labelMat[j]*(alphas[j] - alphaJold)*dataMatrix[j, :]*dataMatrix[j, :].T
                # 设置常数项
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2)/2.0
                alphaPairsChanged += 1
                print("iter: %d, i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
        if(alphaPairsChanged == 0): iter += 1
        else: iter = 0
        print("iteration number: %d" % iter)
    return b, alphas


if __name__ == '__main__':
    dataArr, labelArr = loadDataSet('testSet.txt')
    b, alphas = smoSimple(dataArr, labelArr, 0.6, 0.001, 40)
    print (b)
    print (alphas[alphas>0])            # 数组过滤，只对numpy类型有效
    print (shape(alphas[alphas>0]))     # 支持向量个数
    for i in range(100):
        if alphas[i]>0.0: print (dataArr[i], labelArr[i])