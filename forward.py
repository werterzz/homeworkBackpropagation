import random
import numpy as np
import csv


nodeIn = [1, 0, 1]
firstbias = [0.1, 0.2, 0.3]



def initWeightInLayer(numOfInput, numOfNode):
    array = []
    weight = numOfNode * numOfInput
    i=0
    while i < numOfNode:
        j=0
        temp = []
        # temp.append(1)
        while j < numOfInput:
            temp.append(round(random.random(), 4))
            j = j + 1
        array.append(temp)
        i = i + 1
    return array
# deltaWeight(learningRate, desire, startNode, endNode)
def changeWeightInLayer(weight, learningRate, error, startNode, endNode):

    i=0
    while i < len(weight):
        j=0
        while j < len(weight[i]):
            # print(len(weight))
            # print(len(error))
            if j == 0: weight[i][j] = round(weight[i][j] + -learningRate*(error[i])*endNode[i]*(1-endNode[i])*1, 4)
            else: weight[i][j] = round(weight[i][j] + -learningRate*(error[i])*endNode[i]*(1-endNode[i])*float(startNode[j]), 4)
            j = j + 1
        i = i + 1
    return weight


# weight = initWeightInLayer(3, 3)
# print(weight)


def sumWeight(inputNode, weight):
    i=0
    array = []
    # print(inputNode)
    while i<len(weight):
        j=0
        sum=0
        while j < len(weight[i]):
            if j == 0:
                sum = sum + 1*weight[i][j]
            else:
                sum = sum + float(inputNode[j])*weight[i][j]
            j = j + 1
        # print(sum)
        array.append(round(sum, 4))
        i = i + 1
    return array

# test = sumWeight(nodeIn, firstbias)
# print(test)

def getOutput(sum):
    i=0
    array = []
    while i < len(sum):
        temp = 1/(1+np.exp(-sum[i]))
        array.append(round(temp, 4))
        i = i + 1
    return array

# test2 = getOutput(test)
# print(test2)

def deltaWeight(learningRate, desire, startNode, endNode):
    return -learningRate*(desire-endNode)*endNode*(1-endNode)*startNode

# '# d = 0-0.25, h = 0.25-0.5, s = 0.5-0.75, 0 = 0.75-1.0
def genDesire(desired):
    if desired == 'd ': desired = [1, 0, 0, 0]
    if desired == 'h ': desired = [0, 1, 0, 0]
    if desired == 's ': desired = [0, 0, 1, 0]
    if desired == 'o ': desired = [0, 0, 0, 1]
    return desired

def checkAccuracy(output, desired):
    i = 0
    maxOutput = 0
    index = 0
    while i < len(output):
        if maxOutput < output[i]:
            maxOutput = output[i]
            index = i
        i = i + 1
    i = 0
    maxDesire = 0
    indexDesire = 0
    while i < len(desired):
        if maxDesire < desired[i]:
            maxDesire = desired[i]
            indexDesire = i
        i = i + 1
    if index == indexDesire: return 1
    else: return 0

def sumOnlyWeight(weight):
    i = 0
    sum = 0
    while i < len(weight):
        sum = sum + weight[i]
        i = i + 1
    return sum

def outputError(output, desire):
    error = []
    i = 0
    while i < len(output):
        error.append(desire[i] - output[i])
        i = i + 1
    return error

def calHiddenError(error, weight, numberOfHiddenLayer):
    hiddenError = [0] * numberOfHiddenLayer
    i = 0
    while i < len(hiddenError):
        j = 0
        sum =0
        while j < len(error):
            sumWeight = sumOnlyWeight(weight[j])
            if sumWeight == 0.0: sumWeight = 0.001
            # print("sum " , sumWeight)
            sum = sum + weight[j][i]*error[j]/sumWeight
            j = j + 1
        hiddenError[i] = sum
        i = i + 1
    return hiddenError
            


    


        


# print(deltaWeight(-0.9, 1, 0.332, 0.475))
learningRate = -0.9
desire = [1, 0, 0, 1]
weightLayerTwo = []
weightLayerOne = []

# changeWeightLayerTwo = [[0.1, -0.3, -0.2]]
# changeWeightLayerOne = [[-0.4, 0.2, 0.4, -0.5], [0.2, -0.3, 0.1, 0.2]]


array = []
with open('testing.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        # print(row)
        array.append(row)
# print((array))



outputLayerTwo = [0]
epoch = 100
i = 0
weightLayerOne = initWeightInLayer(27, 10)
weightLayerTwo = initWeightInLayer(10, 4)
bestWeightLayerOne = []
bestWeightLayerTwo = []
accuracyCount = 0
maxAccuracy = 0

while i < epoch:
    j = 1
    accuracyCount = 0
    while j < len(array):
        k = random.randint(1, 325)
        desire = genDesire(array[k][0])
        # print(desire)
        sumLayerOne = sumWeight(array[k], weightLayerOne)
        outputLayerOne = getOutput(sumLayerOne)
        sumLayerTwo = sumWeight(outputLayerOne, weightLayerTwo)
        outputLayerTwo = getOutput(sumLayerTwo)
        # print(outputLayerTwo)
        accuracyCount = accuracyCount + checkAccuracy(outputLayerTwo, desire)
        error = outputError(outputLayerTwo, desire)
        hiddenError = calHiddenError(error, weightLayerTwo, 10)
        weightLayerTwo = changeWeightInLayer(weightLayerTwo, learningRate, error, outputLayerOne, outputLayerTwo)
        weightLayerOne = changeWeightInLayer(weightLayerOne, learningRate, hiddenError, array[k], outputLayerOne)
        j = j + 1
    # print(accuracyCount)
    print("epoch ", i + 1)
    if maxAccuracy < accuracyCount:
        maxAccuracy = accuracyCount
        bestWeightLayerOne = weightLayerOne
        bestWeightLayerTwo = weightLayerTwo

    i = i + 1

# print(weightLayerOne)
# print(weightLayerTwo)
print("maxAccuracy training : ", float(maxAccuracy/326.0))    



array = []
accuracyCount = 0
with open('training.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        # print(row)
        array.append(row)

i = 1
while i < len(array):
    desire = genDesire(array[i][0])
    sumLayerOne = sumWeight(array[i], bestWeightLayerOne)
    outputLayerOne = getOutput(sumLayerOne)
    sumLayerTwo = sumWeight(outputLayerOne, bestWeightLayerTwo)
    outputLayerTwo = getOutput(sumLayerTwo)
    accuracyCount = accuracyCount + checkAccuracy(outputLayerTwo, desire)
    i = i + 1
print("Accuracy testing", float(accuracyCount/198.0))