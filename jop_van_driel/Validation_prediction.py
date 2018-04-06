"""
This script takes in the model generated in Processing.py 
and uses it to generate predictions for the validation data. 

trainfeatlist and testfeatlist need to be saved from the lists of the same name in Processing.py
to allow features to be correctly synchronized
"""

import csv
import pandas
import xgboost as xgb
import numpy
from numpy import array


#Define file locations
dataloc = '.\\validreduced.csv'
trainlist = '.\\trainfeatlist.csv'
testlist = '.\\testfeatlist.csv'
modelloc = '.\\0005.model'
outloc = ".\\validpreds4.csv"

#Import data and encode
data = []

with open(dataloc, newline='') as csvfile:
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
validfeatlist = list(df.columns.values)

#Import featlists for test and training data to synchronize validation data to these
trainfeatlist = []
with open(trainlist, newline='') as csvfile:
     spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
     for row in spamreader:
         trainfeatlist.append(row)

testfeatlist = []
with open(testlist, newline='') as csvfile:
     spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
     for row in spamreader:
         testfeatlist.append(row)
         
for i in range(0, len(trainfeatlist[0])):
    if not trainfeatlist[0][i] in validfeatlist:
        df[trainfeatlist[0][i]] = 0
        
for i in range(0, len(testfeatlist[0])):
    if not testfeatlist[0][i] in validfeatlist:
        df[testfeatlist[0][i]] = 0

df.reindex_axis(sorted(df.columns), axis=1)


#Import model
bst = xgb.Booster({'nthread': 4})  # init model
bst.load_model(modelloc)  # load data

dtest = xgb.DMatrix(df.as_matrix())
# make prediction
preds = bst.predict(dtest)
#Save predictions
numpy.savetxt(outloc, preds, delimiter=",")