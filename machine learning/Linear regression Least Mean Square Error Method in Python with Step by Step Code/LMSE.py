# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# importing modules
import numpy
import pandas
from IPython.display import display


# %%
# reading data
myData = pandas.read_csv("ques1Data.csv")
myData2 = pandas.read_csv("ques1Data.csv")

display(myData)


# %%
# testing .iteritems()
for columnName, columnData in myData.iteritems():
    print(columnName)
    for i in columnData:
        print(i)


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
    
    for j in myData.iloc[i][1:]:
        templist.append(j)
        
    xMat.append(templist)
    
    
xMat = numpy.array(xMat)

display(xMat)


# %%
# transpose X 
xMat_transpose = xMat.transpose()

display(xMat_transpose)


# %%
# beta = ( (X_transpose*X) ^ -1 ) * X_transpose * y

# inverse ( X.T * X )
inversed = numpy.linalg.inv(numpy.dot(xMat_transpose,xMat))

display(inversed)


# %%
# intermidiate beta
tempBeta = numpy.dot(inversed , xMat_transpose)

display(tempBeta)


# %%
# beta 
y = numpy.array(myData["Y"])

beta = numpy.dot(tempBeta , y)
display(beta)


# %%
# function to scale by the y predicted value from normalized form to simple form
def scaleBackYPredicted(originalY , ypredicted):
    maxOriginalY = max(originalY)
    minOriginalY = min(originalY)

    tempList = []

    for j in range(len(ypredicted)):
        yScaled = ( ((ypredicted[j] - 0) / (1 - 0)) * (maxOriginalY - minOriginalY) ) + minOriginalY
        tempList.append(yScaled)
    
    return tempList



# function to predict the y based on test data x
# x is the new input to predict y
# x should be a pandas data frame type
def hypothesisFunction(beta ,  x):
    
    x = numpy.array(x)
    
    # y = b0 + b1*x1 + b2*x2 +         + bk*xk
    
    yPredicted = beta[0]
    
    for i in range(1 , len(beta)):
        yPredicted = yPredicted + ( beta[i] * x[i-1] )
        
    return yPredicted


# %%
# function to normalize the new test data based on original data
# here original data min max are used to normalize the data
def returnNormalisedTestData(originalData , testData):

    for columnName, columnData in testData.iteritems():

        maxI = max(originalData[columnName])
        minI = min(originalData[columnName])

        tempList = []

        for j in range(len(columnData)):
            xStar = ( ((columnData[j] - minI) / (maxI - minI)) * (1 - 0) ) + 0
            tempList.append(xStar)
        
        testData[columnName] = tempList

    return testData


# %%
# test data 
testData = [[[3000]] , [[2000]] , [[1500]]]


# convert test data to data frame
testData = pandas.DataFrame([[3000] , [2000] , [1500]] , columns = ["squareFeet"])

display(testData)


# normalize testData
testDataNormal = returnNormalisedTestData(myData2 , testData)

display(testData)


yPredicted = []

# predict y using hypothesis function
for i in testDataNormal.index:
    yPredicted.append(hypothesisFunction(beta , testDataNormal.iloc[i]))

display(yPredicted)

# scale back the predicted value 
yPredicted = scaleBackYPredicted(myData2["Y"] , yPredicted)

for i in range(len(yPredicted)):
    print("price for plot {} = {}".format(testData.iloc[i].tolist() , yPredicted[i]))


# %%
# Code by harshnative
# Email - harshnative@gmail.com
# Github - https://github.com/harshnative


# %%



