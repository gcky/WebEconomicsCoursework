"""
This script performs the main training for my bidding strategy using a reduced dataset to reduce memory needs
The predictions on the test set as well as the model generated are both saved for later use
"""

import csv
import pandas
import xgboost as xgb
import numpy
from numpy import array

testloc = '.\\testreduced.csv'
trainloc = '.\\reduced3.csv'
modelloc = '.\\0005.model'
outloc = ".\\preds5.csv"

data = []

#Load in test data first as columns of one hot encoding are needed later
with open(testloc, newline='') as csvfile:
     spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
     for row in spamreader:
         data.append(row[0].split(',', maxsplit = 20))
         del row

#clean up memory and remove first row which contains headers         
spamreader = None
del data[0]

#convert to array
data = array(data)

dtype = [('Col1','int32'), ('Col2','float32'), ('Col3','float32')]
index = ['Row'+str(i) for i in range(1, len(data)+1)]

#convert to pandas dataframe
df = pandas.DataFrame(data, index=index)
data = None

#one hot encode
df = pandas.get_dummies(df)
#Store list of features/columns
testfeatlist = list(df.columns.values)
#clear memory
df = None

print("pre reading done")


data = []

#load in training data
with open(trainloc, newline='') as csvfile:
     spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
     for row in spamreader:
         data.append(row[0].split(',', maxsplit = 20))
         del row
         
spamreader = None
del data[0]
print("Import done")

data = array(data)

dtype = [('Col1','int32'), ('Col2','float32'), ('Col3','float32')]
index = ['Row'+str(i) for i in range(1, len(data)+1)]

#split into features and labels and convert to dataframes
featdf = pandas.DataFrame(data[:,1:13], index=index)
labeldf = pandas.DataFrame(data[:, 0], index=index)
data = None

#one hot encode
featdf = pandas.get_dummies(featdf, sparse=True)
labeldf = pandas.get_dummies(labeldf, sparse=True)

#ensure no features/columns in test data are not also present here
trainfeatlist = list(featdf.columns.values)

for i in range(0, len(testfeatlist)):
    if not testfeatlist[i] in trainfeatlist:
        featdf[testfeatlist[i]] = 0

#sort columns        
featdf.reindex_axis(sorted(featdf.columns), axis=1)

print("encoding done")

#Convert dataframes to arrays and import into format xgboost uses
feats = featdf.as_matrix()
featdf = None
labels = labeldf.as_matrix()[:, 1]
labeldf = None

dtrain = xgb.DMatrix(feats, labels)
feats = None
labels = None

# specify parameters via map
param = {'max_depth':15, 'silent':0, 'objective':'binary:logistic', 'nthread':6}
num_round = 10

#Train and save model
bst = xgb.train(param, dtrain, num_round)
bst.save_model(modelloc)

#Clean up and notify that training is complete
dtrain = None
print("Training done")


data = []
#Open test data again for getting predictions
with open(testloc, newline='') as csvfile:
     spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
     for row in spamreader:
         data.append(row[0].split(',', maxsplit = 20))
         del row
         
spamreader = None
del data[0]

data = array(data)

dtype = [('Col1','int32'), ('Col2','float32'), ('Col3','float32')]
index = ['Row'+str(i) for i in range(1, len(data)+1)]

df = pandas.DataFrame(data, index=index)
data = None

df = pandas.get_dummies(df)

#ensure columns/features are matched again
testfeatlist = list(df.columns.values)
for i in range(0, len(trainfeatlist)):
    if not trainfeatlist[i] in testfeatlist:
        df[trainfeatlist[i]] = 0

df.reindex_axis(sorted(df.columns), axis=1)


dtest = xgb.DMatrix(df.as_matrix())
# make prediction
preds = bst.predict(dtest)
#Save predictions
numpy.savetxt(outloc, preds, delimiter=",")

