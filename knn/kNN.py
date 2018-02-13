# coding: utf-8

from numpy import array, zeros, tile, shape
import operator
import matplotlib.pyplot as plt

TESTFILE = '/Users/peilei/MyGitHub/machinelearninginaction/Ch02/datingTestSet.txt'


def classify0(inX, dataSet, labels, k):
    """
    inX：用于分类的输入向量
    dataSet：训练集(行数与labels相同)
    labels：标签向量(行数与dataSet相同)
    k:选择最近邻居的数目
    """
    dataSetSize = dataSet.shape[0]   # Tuple of array dimensions
    # 计算距离
    diffMat = tile(inX, (dataSetSize,1)) - dataSet    # Construct an array by repeating A the number of times given by reps
    print('diffMat:', diffMat)
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)    # Sum of array elements over a given axis
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    # 选择距离最小的k个点
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # 排序
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    # 得到文本行数
    numberOfLines = len(arrayOLines)
    # 创建NumPy矩阵
    # 创建一个以0填充的3列矩阵,zeros:Return a new array of given shape and type, filled with zeros.
    returnMat = zeros((numberOfLines, 3)) 
    classLabelVector = []
    index = 0
    # 解析文件数据到列表
    for line in arrayOLines:
        line = line.strip() 
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(listFromLine[-1])
        index += 1
    return returnMat, classLabelVector


def autoNorm(dataSet):
    # 归一化特征值
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))
    return normDataSet, ranges, minVals


def datingClassTest():
    # 测试函数
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix(TESTFILE)
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("the classifier came back with:%s, the real answer is:%s"% (classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]: errorCount += 1.0
    print("the total error rate is:%f" % (errorCount/float(numTestVecs)))


def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input("percentage of time spent playing video games?"))
    ffmiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per yesr?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffmiles, percentTats, iceCream])
    classifierResult = classify0((inArr-minVals)/ranges, normMat, datingLabels, 3)
    print("You will probably like this person: ", resultList[classifierResult - 1])


def show_graph(mat):
    fig = plt.figure()                  # Creates a new figure.
    ax = fig.add_subplot(111)
    ax.scatter(mat[:, 1], mat[:, 2])    # Make a scatter plot of x vs y.Marker size is scaled by s and marker color is
    # mapped to c. ^:按位异或运算符
    plt.show()


if __name__ == '__main__':
    returnMat, classLabelVector = file2matrix(TESTFILE)
    # print(returnMat)
    # print(classLabelVector)
    show_graph(returnMat)
    datingClassTest()
