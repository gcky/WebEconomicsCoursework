"""
This script serves to generate bids for the multiagent setting 
It takes in the predictions on either the validation or test sets and outputs a list of bids.
"""
import csv
import numpy as np
from numpy import array

#Locations of all the relevant files
dataloc = '.\\test.csv'
predsloc = '.\\kwok_test_preds.csv'
outloc = ".\\testing_bidding_price.csv"


#import data
data = []

with open(dataloc, newline='') as csvfile:
     spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
     for row in spamreader:
         data.append(row[0].split(',', maxsplit = 24))
         
spamreader = None     
del data[0]


#Import predictions
preds = []

with open(predsloc, newline='') as csvfile:
     spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
     for row in spamreader:
         preds.append(float(row[0].split(',')[1]))
         
spamreader = None

preds = array(preds)

#Define cutoff above which impressions get bid on
cutoff = np.percentile(preds, 77)
        
#Define parameters for simple bidding strategy. In theory these can be replaced by a single constant bid
a = 200
b = 580

bids = [['bidid', 'bidprice']]

n = 0
paid = 0
clicks = 0

#Iterate over all impressions
for i in range(len(preds)):
    #If the prediction is sufficient place a bid
    if preds[i] > cutoff:
        bid = a*preds[i] + b
        bids.append([data[i][2], bid])
        #Calculate some statistics to be used for finding optimal parameters
        """
        if bid > float(data[i][20]):
            n += 1
            paid += float(data[i][21])
            clicks += float(data[i][0])
        """
    #Otherwise bid 0 to ensure all impressions are accounted for
    else:
        bids.append([data[i][2], 0])
        
#Print statistics
"""        
print(clicks/n)
print(n)
print(clicks)
print(paid/n)
print(paid)
print((paid/1000)/clicks)
"""
#Save bids
with open(outloc,'w') as resultFile:
    wr = csv.writer(resultFile, dialect='excel')
    wr.writerows(bids)

      