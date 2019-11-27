import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from apyori import apriori
import time
start_time = time.time()

#### Input
with open('./data/Airbnb_Cleaned.csv', encoding='utf8') as csvfile:
    data = pd.read_csv(csvfile)
# collect useful columns
x = data.loc[:,['price','property_type','room_type','accommodates','bathrooms',
'bedrooms','beds']]
x.describe()
x.dtypes


#### Clean price column
# bin the column
bins = [0,50,100,150,200,1100]
labels = ['Price_1','Price_2','Price_3','Price_4','Price_5']
x['Price bin'] = pd.cut(x['price'], bins=bins,labels = labels)
# merge to a new dataframe
newdata = pd.DataFrame()
newdata['Price']=x['Price bin']

#### Merge property type and room type to the new datafram
#newdata['Property_type']=x['property_type']
#newdata['Room_type']=x['room_type']

#### Clean accomodates column
# bin the column and merge
bins = list(range(0,21,1))
labels = ['Accommodate_1','Accommodate_2','Accommodate_3','Accommodate_4',
'Accommodate_5','Accommodate_6','Accommodate_7','Accommodate_8',
'Accommodate_9','Accommodate_10','Accommodate_11','Accommodate_12',
'Accommodate_13','Accommodate_14','Accommodate_15','Accommodate_16',
'Accommodate_17','Accommodate_18','Accommodate_19','Accommodate_20']
newdata['Accomodate'] = pd.cut(x['accommodates'], bins=bins,labels = labels)

#### Clean bathrooms column
# bin the column and merge
bins = list(range(0,21,1))
labels = ['Bathroom_1','Bathroom_2','Bathroom_3','Bathroom_4',
'Bathroom_5','Bathroom_6','Bathroom_7','Bathroom_8','Bathroom_9',
'Bathroom_10','Bathroom_11','Bathroom_12','Bathroom_13',
'Bathroom_14','Bathroom_15','Bathroom_16','Bathroom_17',
'Bathroom_18','Bathroom_19','Bathroom_20']
newdata['Bathrooms'] = pd.cut(x['bathrooms'], bins=bins,labels = labels)

#### Clean bedroom column
# bin the column and merge
bins = list(range(0,21,1))
labels = ['Bedroom_1','Bedroom_2','Bedroom_3','Bedroom_4',
'Bedroom_5','Bedroom_6','Bedroom_7','Bedroom_8','Bedroom_9',
'Bedroom_10','Bedroom_11','Bedroom_12','Bedroom_13',
'Bedroom_14','Bedroom_15','Bedroom_16','Bedroom_17',
'Bedroom_18','Bedroom_19','Bedroom_20']
newdata['Bedrooms'] = pd.cut(x['bedrooms'], bins=bins,labels = labels)

#### Clean Bed column
# bin the column and merge
bins = list(range(0,21,1))
labels = ['Bed_1','Bed_2','Bed_3','Bed_4',
'Bed_5','Bed_6','Bed_7','Bed_8','Bed_9',
'Bed_10','Bed_11','Bed_12','Bed_13',
'Bed_14','Bed_15','Bed_16','Bed_17',
'Bed_18','Bed_19','Bed_20']
newdata['Beds'] = pd.cut(x['beds'], bins=bins,labels = labels)

#### Data proprocessing
# this part might take about 20 minutes to run, but once it has been done,
# apriori should run very quitely with the dataset 'record'
print('Processing data:')
records_1 = []
for i in range(0, len(newdata)):
	records_1.append([str(newdata.values[i,j]) for j in range(0,5)])
print('Done!')

# min_support = 0.025, min_confidence = 0.1, min_lift=2 ######################
print('Processing algorithm:')
rules = apriori(records_1, min_support = 0.025, min_confidence = 0.1, min_lift=2,
	min_length=2)
results = list(rules)
print('Done!')
print("==================================")
print('Number of results: ', len(results))
#### print out results
for item in results:
    pair = item[0]
    items = [x for x in pair]
    print("Rule: "+ items[0] + '->' + items[1])
    print("Sup: " + str(item[1]))
    print("Conf: " + str(item[2][0][2]))
    print("Lift: " + str(item[2][0][3]))
    print("==================================")


# min_support = 0.05, min_confidence = 0.1, min_lift=2 ######################
print('Processing algorithm:')
rules = apriori(records_1, min_support = 0.05, min_confidence = 0.1, min_lift=2,
	min_length=2)
results = list(rules)
print('Done!')
print("==================================")
print('Number of results: ', len(results))
#### print out results
for item in results:
    pair = item[0]
    items = [x for x in pair]
    print("Rule: "+ items[0] + '->' + items[1])
    print("Sup: " + str(item[1]))
    print("Conf: " + str(item[2][0][2]))
    print("Lift: " + str(item[2][0][3]))
    print("==================================")
    
    
# min_support = 0.075, min_confidence = 0.1, min_lift=2 ######################
print('Processing algorithm:')
rules = apriori(records_1, min_support = 0.075, min_confidence = 0.1, min_lift=2,
	min_length=2)
results = list(rules)
print('Done!')
print("==================================")
print('Number of results: ', len(results))
#### print out results
for item in results:
    pair = item[0]
    items = [x for x in pair]
    print("Rule: "+ items[0] + '->' + items[1])
    print("Sup: " + str(item[1]))
    print("Conf: " + str(item[2][0][2]))
    print("Lift: " + str(item[2][0][3]))
    print("==================================")



print("--- Run time: %s seconds ---" % (time.time() - start_time))









