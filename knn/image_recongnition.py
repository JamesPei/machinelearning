# -*- coding:utf-8 -*-

from numpy import array, zeros
from os import listdir

from knn.kNN import classify0

TRAINDIR = '/Users/peilei/MyGitHub/machinelearninginaction/Ch02/digits/trainingDigits/'
TESTDIR = '/Users/peilei/MyGitHub/machinelearninginaction/Ch02/digits/testDigits/'


def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect


def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir(TRAINDIR)
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector(TRAINDIR + fileNameStr)
    testFileList = listdir(TESTDIR)
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector(TESTDIR + fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print('the classifier came back with %d, the real answer id: %d' % (classifierResult, classNumStr))
        if classifierResult != classNumStr: errorCount += 1.0
    print("the total number of errors is:%d" % errorCount)
    print("the total error rate is: %f" % (errorCount/float(mTest)))


if __name__ == '__main__':
    vector = img2vector(TRAINDIR + '3_18.txt')
    print(vector[0, 32:63])