# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy
import pandas
from IPython.display import display
import math
import sqlitewrapper

# data base for caching of beta's
dbObj = sqlitewrapper.SqliteCipher(password="none")

try:
    dbObj.createTable("cache" , [["DataBase" , "TEXT"] , ["cachedBeta" , "LIST"] , ["maxIteration" , "TEXT"] , ["alpha" , "TEXT"]])
except ValueError:
    pass


# %%
myData = pandas.read_csv("Iris.csv")

# remove any null values
myData.dropna()

display(myData)


# %%
yLabel = myData["Species"]

# select important cols only
myData = myData[["SepalLengthCm" , "SepalWidthCm" , "PetalLengthCm" , "PetalWidthCm"]]
display(myData)

display(yLabel)


# %%
# normalize data using min max normalization
for columnName, columnData in myData.iteritems():
    maxI = max(columnData)
    minI = min(columnData)

    tempList = []

    for j in range(len(columnData)):
        xStar = ( ((columnData[j] - minI) / (maxI - minI)) * (1 - 0) ) + 0
        tempList.append(xStar)
        
    myData[columnName] = tempList
    
display(myData)


# %%
# convert the X to matrix
# [
#     1 X11        X1k
#     1 X21        X2k
#     1 X22        X3k
    
    
#     1 Xn1        Xnk
# ]

xMat = []

for i in myData.index:
    templist = [1]
    
    for j in myData.iloc[i]:
        templist.append(j)
        
    xMat.append(templist)
    
    
xMat = numpy.array(xMat)

display(xMat)


# %%
# number of beta's
kValue = len(xMat[0])

print(kValue)

# number of rows in data
nValue = len(xMat)

print(nValue)


# %%
# number of unique classes
yLabelSet = list(set(yLabel))
yLabelSet = sorted(yLabelSet)

print(yLabelSet)

lenYLabelSet = len(yLabelSet)

print(lenYLabelSet)


# %%
# generate Dataset = number of unique classes = U
# dataset 1 -> y = 1 for class 1 and y = 0 for other class in data set
# dataset 2 -> y = 1 for class 2 and y = 0 for other class in data set
# dataset U -> y = 1 for class U and y = 0 for other class in data set


listOfDataSets = []

# DataFrame.copy(deep=True)

for i in range(lenYLabelSet):
    newMyData = myData.copy(deep=True)

    yCol = []

    for j in newMyData.index:
        if(yLabel.iloc[j] == yLabelSet[i]):
            yCol.append(1)
        else:
            yCol.append(0)

    newMyData["Y"] = yCol

    listOfDataSets.append(newMyData)


for i in listOfDataSets:
    display(i)


# %%
useCaching = True


# %%
# %%
# BetaK's for each dataSet

# betaK's = 

# beta[j] = beta[j] - alpha * slope

# slope =  ( 1/n ) * Submission 1 to n {(f[xi] - yi) * xij}

# f[xi] = 1 / 1 + e^z

# z = -1 * (beta0 + beta1*Xi1 +             + betak*Xik)


# getDataSets From cache
_ , dataSetsFromCache = dbObj.getDataFromTable("cache" , omitID=True)


betasList = []


for dataSetCount , dataSet in enumerate(listOfDataSets):

    alpha = 0.2

    # max iterations to find the beta
    maxIteration = 50000

    # accuracy of beta needed
    betaErrorTolerance = 5

    is_dataSetInCache = False
    betaCache = []

    for dataSetFromCache in dataSetsFromCache:
        if(str(dataSet) == str(dataSetFromCache[0])):
            if((dataSetFromCache[2] == str(maxIteration)) and (dataSetFromCache[3] == str(alpha)) and (useCaching)):
                is_dataSetInCache = True
                betaCache = dataSetFromCache[1]

    if(is_dataSetInCache):
        print("\n\nusing data set from cache for class {}. On {} / {}".format(yLabelSet[dataSetCount] , dataSetCount , lenYLabelSet))
        betasList.append(betaCache)
        print()
        print(betaCache)

    else:

        y = dataSet["Y"]

        print("\n\nprocessing data set for class {}. On {} / {}".format(yLabelSet[dataSetCount] , dataSetCount , lenYLabelSet))


        # init bk's as zero
        # init tempk's as zero 
        betaKs = [0 for _ in range(kValue)]
        tempKs = [0 for _ in range(kValue)]
        slopeKs = [0 for _ in range(kValue)]


        for iteration in range(maxIteration):
            
            breakLoop = False
            
            for j in range(kValue):
                slope = 1 / nValue
                submission = 0
                k = 1


                for i in range(nValue):
                    innerSubmission = betaKs[0]

                    for k in range(1 , kValue):
                        innerSubmission = innerSubmission + (betaKs[k] * xMat[i][k])

                    innerSubmission = innerSubmission * -1

                    innerSubmission = 1 / (1 + math.exp(innerSubmission))

                    innerSubmission = innerSubmission - y[i]

                    innerSubmission = innerSubmission * xMat[i][j]

                    submission = submission + innerSubmission

                slope = submission * slope

                tempKs[j] = betaKs[j] - (alpha * slope)

                slopeKs[j] = slope


            betaTolerancedReached = 0

            for a,b in zip(betaKs , tempKs):
                if(round(a , betaErrorTolerance) == round(b , betaErrorTolerance)):
                    breakLoop = True
                else:
                    breakLoop = False
                    betaTolerancedReached = round(b , betaErrorTolerance) - round(a , betaErrorTolerance)

            print("\ron {} , betaToleranced = {}".format(iteration , betaTolerancedReached) , end="")

            if(breakLoop):
                print("break on" , iteration)
                break

                
            # assign bj = tempj
            for j in range(kValue):
                betaKs[j] = tempKs[j]
        
        print()
        print(betaKs)
        
        betasList.append(betaKs)

        dbObj.insertIntoTable("cache" , [str(dataSet) , betaKs , str(maxIteration) , str(alpha)])



print("\n\n")

for i in betasList:
    print(i)
    print()


# %%
# function to predict y based on new x input
# x is the new input to predict y
# x must be a data frame type
def hypothesisFunction(beta ,  x):
    
    x = numpy.array(x)
    
    # y = 1 / 1 + e^((b0 + b1*x1 + b2*x2 +         + bk*xk)*-1)
    
    yPredicted = beta[0]
    
    for i in range(1 , len(beta)):
        yPredicted = yPredicted + ( beta[i] * x[i-1] )

    yPredicted = yPredicted * -1
    yPredicted = 1 / (1 + math.exp(yPredicted))
        
    return yPredicted


# function to find the max value of yPredicted and assign the label
# returns a list containing [yPredicted , label]
def maxHypothesisFunction(betaList , yLabelSet , x):

    yPredictedList = []

    for i,j in zip(yLabelSet , betaList):

        yPredicted = hypothesisFunction(j , x)

        yPredictedList.append([yPredicted , i])

    yPredictedList = sorted(yPredictedList , key=lambda x:x[0] , reverse=True)

    # returning max yPredicted
    return yPredictedList[0]


# %%
# function to normalize the new test data based on original data
# here original data min max are used to normalize the data
def returnNormalisedTestData(originalData , testData):

    for columnName, columnData in testData.iteritems():

        maxI = max(originalData[columnName])
        minI = min(originalData[columnName])

        xStar = ( ((columnData - minI) / (maxI - minI)) * (1 - 0) ) + 0

        testData[columnName] = xStar

    return testData


# %%
myData = pandas.read_csv("Iris.csv")

# remove any null values
myData.dropna()

display(myData)

confusionMatrixList = []

print()

# build confusion matrix for each class
for yLabelSetI in yLabelSet:
    print("testing for {}".format(yLabelSetI))
    TP = 0
    TN = 0
    FN = 0
    FP = 0

    # traverse data set 
    for i in myData.index:
        features = myData.iloc[i][1:-1]
        features = returnNormalisedTestData(myData[["SepalLengthCm" , "SepalWidthCm" , "PetalLengthCm" , "PetalWidthCm"]] , features)
        predictedLabel = maxHypothesisFunction(betasList , yLabelSet , features)
        actualValue = myData.iloc[i].Species

        # was True , predicted True
        if((actualValue == yLabelSetI) and (predictedLabel[1] == yLabelSetI)):
            TP = TP + 1
        
        # was false , predicted True
        elif(not((actualValue == yLabelSetI)) and (predictedLabel[1] == yLabelSetI)):
            FP = FP + 1
        
        # was True , predicted False
        elif((actualValue == yLabelSetI) and (not(predictedLabel[1] == yLabelSetI))):
            FN = FN + 1
        
        # was False , predicted False
        elif((not(actualValue == yLabelSetI)) and (not(predictedLabel[1] == yLabelSetI))):
            TN = TN + 1

    accuracy_confusionMatrix = (TP + TN) / (TP + TN + FP + FN)
    precision_confusionMatrix = TP / (TP + FP)

    confusionMatrixList.append([yLabelSetI , [[TP , FP] , [FN , TN]] , accuracy_confusionMatrix , precision_confusionMatrix])

print()           
                
for i in confusionMatrixList:
    print("Stats for {}".format(i[0]))

    print("confusion matrix  = ")

    display(i[1])

    print("accuracy = {}".format(i[2]))
    print("precision = {}".format(i[3]))

    print("\n")


