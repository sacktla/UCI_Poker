"""
@author: Jesus Zaragoza
@project: UCI poker guess hand
"""
##First step is to create a data frame using pandas so that we 
##can do sorting of columns

import pandas as pd

#Create Data Frame from urllib

import urllib2

webData = urllib2.urlopen('https://archive.ics.uci.edu/ml/machine-learning-databases/poker/poker-hand-training-true.data')

#Create header

header = []
suitCounter = 0
rankCounter = 0
for x in range(10):
   if x%2==0:
	suitCounter += 1
	header.append(str(suitCounter)+"_Suit")
   else:
	rankCounter += 1
	header.append(str(rankCounter)+"_Rank")
header.append("Play")

#Create Data frame with headers
trainingDF = pd.read_csv(webData,header=None)
trainingDF.columns = header
print "Printing Training Data Frame head"
print trainingDF.head()
webData.close()


'''Now the idea is to sort pairs by ranking'''
##Loop through the frame and create a dictionary that will help you keep the pair relationship
##Sort the dictionary and repopulate the Data Frame.

import operator#needed for sorting dictionary
for i in range(len(trainingDF.index)):
	tempList = trainingDF.loc[i].tolist()
	tempDict = {"A-"+str(tempList[0]):tempList[1],"B-"+str(tempList[2]):tempList[3],"C-"+str(tempList[4]):tempList[5],"D-"+str(tempList[6]):tempList[7],"E-"+str(tempList[8]):tempList[9]}
	sortedDict = sorted(tempDict.items(),key=operator.itemgetter(1))
	trainingDF.loc[i]["1_Suit"] = sortedDict[0][0].split("-")[1]
	trainingDF.loc[i]["1_Rank"] = sortedDict[0][1]
	trainingDF.loc[i]["2_Suit"] = sortedDict[1][0].split("-")[1]
	trainingDF.loc[i]["2_Rank"] = sortedDict[1][1]
	trainingDF.loc[i]["3_Suit"] = sortedDict[2][0].split("-")[1]
	trainingDF.loc[i]["3_Rank"] = sortedDict[2][1]
	trainingDF.loc[i]["4_Suit"] = sortedDict[3][0].split("-")[1]
	trainingDF.loc[i]["4_Rank"] = sortedDict[3][1]
	trainingDF.loc[i]["5_Suit"] = sortedDict[4][0].split("-")[1]
	trainingDF.loc[i]["5_Rank"] = sortedDict[4][1]

#Now you should have a sorted trainingDF. You can now convert it to a numpy array.
trainingNpArray = trainingDF.as_matrix(columns=None)
print "Printing first 10 rows of Training Numpy Array"
print trainingNpArray[0:10]

#Now do line 13-51 for the test data and then run the model
webData = urllib2.urlopen('https://archive.ics.uci.edu/ml/machine-learning-databases/poker/poker-hand-testing.data')
testingDF = pd.read_csv(webData,header=None,nrows=10000) #nrows allows you to select a n number of rows from data. Useful for a million rows.
testingDF.columns = header
webData.close()

print "Printing Testing Data Frame head"
print testingDF.head()

for i in range(len(testingDF.index)):
	tempList = testingDF.loc[i].tolist()
	tempDict = {"A-"+str(tempList[0]):tempList[1],"B-"+str(tempList[2]):tempList[3],"C-"+str(tempList[4]):tempList[5],"D-"+str(tempList[6]):tempList[7],"E-"+str(tempList[8]):tempList[9]}
        sortedDict = sorted(tempDict.items(),key=operator.itemgetter(1))
        testingDF.loc[i]["1_Suit"] = sortedDict[0][0].split("-")[1]
        testingDF.loc[i]["1_Rank"] = sortedDict[0][1]
        testingDF.loc[i]["2_Suit"] = sortedDict[1][0].split("-")[1]
        testingDF.loc[i]["2_Rank"] = sortedDict[1][1]
        testingDF.loc[i]["3_Suit"] = sortedDict[2][0].split("-")[1]
        testingDF.loc[i]["3_Rank"] = sortedDict[2][1]
        testingDF.loc[i]["4_Suit"] = sortedDict[3][0].split("-")[1]
        testingDF.loc[i]["4_Rank"] = sortedDict[3][1]
        testingDF.loc[i]["5_Suit"] = sortedDict[4][0].split("-")[1]
        testingDF.loc[i]["5_Rank"] = sortedDict[4][1]
	
testingNpArray = testingDF.as_matrix(columns=None)
print "Printing first 10 rows of Testing Numpy Array"
print testingNpArray[0:10]

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import metrics
from sklearn.svm import SVC

classifier = SVC()

print trainingNpArray.shape
trainInput = trainingNpArray[:,0:10]
trainOutput = trainingNpArray[:,10]

print "training"
classifier.fit(trainInput,trainOutput)


testInput = testingNpArray[:,0:10]
testOutput = testingNpArray[:,10]

print "Predicting"
predicted = classifier.predict(testInput)
fpr,tpr,thresholds = metrics.roc_curve(testOutput,predicted,pos_label=3)#If you want to increase the pos label you need to increase the number of lines are read.

auc = metrics.auc(fpr,tpr)

print auc






