from math import log
import operator
import ml


def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    # 获得数据集的行数
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        # 正确的结果是每一个样本的最后一列
        if currentLabel not in labelCounts.keys():
            # 如果正确的结果没在类别标签里
            labelCounts[currentLabel] = 0
            # 将该结果加入类别标签，key为标签，value为0
        labelCounts[currentLabel] += 1
        # 记录该标签出现的次数
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        # 对标签集的每一类，出现的次数/模数 = P（该类别被选择的概率）
        shannonEnt -= prob * log(prob, 2)
        # ∑ -p*log(p, 2)
    return shannonEnt

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet,labels


def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        # 对dataSet中的每一行进行循环
        if featVec[axis] == value:
            # 如果每一行的第 axis 个特征为 value
            reducedFeatVec = featVec[:axis]
            # 因为list切片时包前不包后，所以相当于将第axis个特征取出，然后将样本加入到新的list中
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    # 列数-1为特征数
    baseEntropy = calcShannonEnt(dataSet)
    # 计算数据集的香农熵
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        # featList中存放每一行中的第i个特征值
        uniqueVals = set(featList)
        # 相当于排重
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            # 对第 i 个特征的所有值进行数据集划分
            prob = len(subDataSet)/float(len(dataSet))
            # 选到这一类的概率
            newEntropy += prob * calcShannonEnt(subDataSet)
            # 计算这一种划分方式的信息熵
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        # 对于classList的每一个值
        if vote not in classCount.keys():classCount[vote] = 0
        # 如果这个值不存在于classCount的key中，那么创建一个key，其value为0
        # classCount用于计算每一种分类出现的次数
        classCount[vote] += 1
        sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
        # 将类别按values排序字典，返回次数出现最多的类
        # iteritems是python2.x的写法
        return sortedClassCount[0][0]

def createTrees(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    # 取数据集每一行最后一个特征，即分类结果，组成classList
    if classList.count(classList[0]) == len(classList):
        # 如果classList[0]出现的次数等同于classList的长度，即所有元素值相同
        return classList[0]
        # 分类完成，返回结果
    if len(dataSet[0]) == 1:
        # 如果使用完所有的特征值，仍然无法将数据集划分成仅包含一个类别的分组，则返回出现次数最多的类别
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    # 使用最好的特征进行划分
    bestFeatLabel = labels[bestFeat]
    # 最好的特征
    myTree = {bestFeatLabel:{}}
    # 使用bestFeatLabes做key，值为一个空字典
    # myTree
    # 是一个字典，其key为类名，其value为
    del(labels[bestFeat])
    # 已经创建过这个类别了，从labels列表删除
    featValues = [example[bestFeat] for example in dataSet]
    # 取dataSet中最好的特征对应的所有数据
    uniqueVals = set(featValues)
    # 排重，最好的特征中出现了多少种元素
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTrees(splitDataSet(
            dataSet, bestFeat, value
        ), subLabels)
    return myTree
myDat, labels = createDataSet()
trees = createTrees(myDat, labels)
ml.createPlot()