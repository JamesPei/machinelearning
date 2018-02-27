# -*-coding:utf-8-*-
from math import log, exp
from numpy import shape, ones, mat, zeros, array
from numpy.core.umath import sign
from numpy.ma import multiply

from boost import buildStump, stumpClassify
import matplotlib.pyplot as plt

"""
伪代码：
对每次迭代：
    利用buildStump()函数找到最佳的单层决策树
    将最佳单层决策树加入到单层决策树数组
    计算alpha
    计算新的权重向量D
    更新累计类别估计值
    如果错误率等于0.0，退出循环
"""


def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    """
    :param numIt: 迭代次数
    """
    weakClassArr = []
    m = shape(dataArr)[0]
    D = mat(ones((m, 1))/m)     # D是一个概率分布向量，因此其所有元素之和为1.0
    aggClassEst = mat(zeros((m, 1)))    # 记录每个数据点的类别估计累计值
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        print ("D:", D.T)
        alpha = float(0.5*log((1.0-error)/max(error, 1e-16)))   # 本次单层决策树输出结果的权重
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        print ("classEst:", classEst.T)
        # 为下一次迭代计算D
        expon = multiply(-1*alpha*mat(classLabels).T, classEst)
        D = multiply(D, exp(expon))
        D = D/D.sum()
        # 错误率累加计算
        aggClassEst += alpha*classEst
        print "aggClassEst: ", aggClassEst.T
        aggErrors = multiply(sign(aggClassEst)!=mat(classLabels).T, ones((m, 1)))
        errorRate = aggErrors.sum()/m
        print ("total error: ", errorRate, "\n")
        if errorRate == 0.0: break
    return weakClassArr, aggClassEst


# adaboost分类函数
def adaClassify(datToClass, classifierArr):
    """

    :param datToClass: 待分类样例
    :param classifierArr: 多个弱分类器
    :return:
    """
    dataMatrix = mat(datToClass)
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m, 1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'],
                                 classifierArr[i]['thresh'],
                                 classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
        print aggClassEst
    return sign(aggClassEst)


def loadDataSet(fileName):      #general function to parse tab -delimited floats
    numFeat = len(open(fileName).readline().split('\t')) #get number of fields
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat


def plotROC(predStrengths, classLabels):
    """

    :param predStrengths: 分类器预测强度
    :param classLabels:
    :return:
    """
    cur = (1.0, 1.0)
    ySum = 0.0
    numPosClas = sum(array(classLabels) == 1.0)
    yStep = 1/float(numPosClas)
    xStep = 1/float(len(classLabels)-numPosClas)
    # 获取排好序的索引
    sortedIndicies = predStrengths.argsort()
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:
            delX = 0
            delY = yStep
        else:
            delX = xStep
            delY = 0
            ySum += cur[1]
        ax.plot([cur[0], cur[0]-delX], [cur[1], cur[1]-delY], c='b')
        cur = (cur[0]-delX, cur[1]-delY)
    ax.plot([0, 1], [0, 1], 'b--')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve for AdaBoost horse colic detection system')
    ax.axis([0, 1, 0, 1])
    plt.show()
    print "the Area Under the Curve is: ", ySum * xStep


if __name__ == '__main__':
    datArr, labelArr = loadDataSet('horseColicTraining2.txt')
    classifierArray, aggClassEst = adaBoostTrainDS(datArr, labelArr, 10)
    testArr, testLabelArr = loadDataSet('horseColicTest2.txt')
    predication10 = adaClassify(testArr, classifierArray)
    errArr = mat(ones((67, 1)))
    print (errArr[predication10 != mat(testLabelArr).T].sum())
    plotROC(aggClassEst.T, labelArr)