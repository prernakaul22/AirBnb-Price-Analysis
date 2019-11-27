#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 18:48:05 2018

@author: OliverQiu
"""

import pandas as pd
from os import path
import matplotlib.pyplot as plt
from wordcloud import WordCloud

#### import data
data=pd.read_csv('nyc.csv',sep=',')
df=pd.DataFrame({'price':data['price'],'description':data['description'],
                          'zipcode':data['zipcode']})

#### seperate data base on price range (0-200,200-400,400-1000)
low = df[(df['price']<=200) & (df['price'] > 0)]
mid = df[(df['price']<=400) & (df['price'] > 200)]
high = df[df['price'] > 400]

#### clean data

#### store description data into files
low.to_csv('low_description.txt', header=None, index=None, sep=' ',mode='a')
mid.to_csv('mid_description.txt', header=None, index=None, sep=' ',mode='a')
high.to_csv('high_description.txt', header=None, index=None, sep=' ',mode='a')


#### make wordcloud
### low
d = path.dirname(__file__)
Rawfilename="low_description.txt"
#read text
text = open(path.join(d, Rawfilename)).read()
#wordcloud
wordcloud = WordCloud().generate(text)
# Open a plot of the generated image.
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

### mid
d = path.dirname(__file__)
Rawfilename="mid_description.txt"
#read text
text = open(path.join(d, Rawfilename)).read()
#wordcloud
wordcloud = WordCloud().generate(text)
# Open a plot of the generated image.
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

### high
d = path.dirname(__file__)
Rawfilename="high_description.txt"
#read text
text = open(path.join(d, Rawfilename)).read()
#wordcloud
wordcloud = WordCloud().generate(text)
# Open a plot of the generated image.
plt.imshow(wordcloud)
plt.axis("off")
plt.show()