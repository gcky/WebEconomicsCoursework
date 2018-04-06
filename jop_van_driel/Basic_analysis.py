"""
This script performs the basic analysis of the dataset for use on question 1
"""


import csv
import numpy as np
import matplotlib.pyplot as plt


#Import data
data = []

dataloc = '.\\train.csv'

with open(dataloc, newline='') as csvfile:
     spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
     for row in spamreader:
         data.append(row[0].split(',', maxsplit = 24))
         
spamreader = None     
del data[0]


#count rows
n = 0;
for row in data:
    n = n+1


#initialize variables for analysis
clickcount = 0
pricetotal = 0
bidtotal = 0
clickpricetotal = 0
hourcount = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
             [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
daycount = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
citycount = np.zeros((500, 2))
minbid = 0
maxbid = 0

#calculate statistics from data
for row in data:
    clickcount = clickcount + float(row[0])
    pricetotal = pricetotal + float(row[21])
    bidtotal = bidtotal + float(row[20])
    if row[0] == "1":
        clickpricetotal = clickpricetotal + float(row[21])
    if row[1] != "null":
        daycount[int(row[1])-1] [0] = daycount[int(row[1])-1] [0] + 1
        daycount[int(row[1])-1] [1] = daycount[int(row[1])-1] [1] + float(row[0])
    if row[2] != "null":
        hourcount[int(row[2])-1] [0] = hourcount[int(row[2])-1] [0] + 1
        hourcount[int(row[2])-1] [1] = hourcount[int(row[2])-1] [1] + float(row[0])
    if row[8] != "null":
        citycount[int(row[8])-1] [0] = citycount[int(row[8])-1] [0] + 1
        citycount[int(row[8])-1] [1] = citycount[int(row[8])-1] [1] + float(row[0]) 
    if float(row[20]) < minbid:
        minbid = float(row[20])
    if float(row[20]) > maxbid:
        maxbid = float(row[20])

#Convert totals to averages and counts to CTRs    
CTR = clickcount/n
avgPrice = pricetotal/n
avgBid = bidtotal/n
avgcpc = clickpricetotal/clickcount

cityCTR = []
for row in citycount:
    if row[0] != 0:
        cityCTR.append(row[1]/row[0])
    
hourCTR = []    
for i in range(len(hourcount)):
    hourCTR.append([i, hourcount[i][1]/hourcount[i][0]])
    
dayCTR = []    
for i in range(len(daycount)):
    dayCTR.append([i, daycount[i][1]/daycount[i][0]])
        
dayCTR = np.transpose(np.array(dayCTR))    
hourCTR = np.transpose(np.array(hourCTR)) 
cityCTR = np.array(cityCTR)

            
#Generate plots
plt.hist(cityCTR, bins=50)
plt.show()
plt.close()

plt.plot(hourCTR[0], hourCTR[1])
plt.show()
plt.close()

plt.plot(dayCTR[0], dayCTR[1])


#Print statistics
print("n: " + str(n))
print("CTR: " + str(CTR))
print("avgPrice: " + str(avgPrice))
print("avgBid: " + str(avgBid))
print("avgclickprice: " +str(avgcpc))
print("min bid: " + str(minbid))
print("max bid: " + str(maxbid))

