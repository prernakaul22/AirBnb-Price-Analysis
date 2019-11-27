#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 16:26:27 2018

@author: stevechen
"""
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def Get_Data(filename):
    data = pd.read_csv(filename,sep=',')
    my_df = pd.DataFrame({'price':data['price'],'property_type':data['property_type'],
                          'room_type':data['room_type'],'accommodates':data['accommodates'],
                          'bathrooms':data['bathrooms'],'bedrooms':data['bedrooms'],
                          'beds':data['beds'],'bed_type':data['bed_type'],
                          'guests_included':data['guests_included'],
                          'number_hotels':data['number_hotels'],'hotel_meanprice':data['hotel_meanprice'],
                          'num_res':data['num_res']})
    return (my_df)


# Transform Categorical variables into dummies:     
def Category_to_Dummy(data,category_var_list):
    for i in category_var_list:
        # Create dummies variables and concatenate them
        dummies=pd.get_dummies(data[i])
        data=pd.concat([data,dummies],axis=1)
        
        # Drop the categorical variable
        data=data.drop([i],axis=1)
        
    return (data)

def Corr_HeatMap(data):
    # Correlation Matrix 
    f, ax = plt.subplots(figsize=(10, 6))
    corr = data.corr()
    hm = sns.heatmap(round(corr,2), annot=True, ax=ax, cmap="coolwarm",fmt='.2f',
                 linewidths=.05)
    f.subplots_adjust(top=0.93)
    t= f.suptitle('Price Attributes Correlation', fontsize=14)

# Glance at the data   
def Glancing_Data(data):
    pd.set_option('display.max_columns', None)
    print(data.info())
    print(data.describe(include='all'))
    


def Boxplot(data):
     sns.set(style="whitegrid")
     tips = sns.load_dataset("tips")
     
     #ax = sns.boxplot(x=data["price"])
     
     ax = sns.boxplot(x="room_type", y="price", data=data)

import statsmodels.api as sm
def Linear_model(data):
    X = data.loc[:,['bathrooms','Private room','Entire home/apt',
                    'Shared room','hotel_meanprice','accommodates','guests_included',
                    'beds','num_res']]
    ## mpg is our dependent variable
    Y = data['price']
    # create the model
    mod1res = sm.OLS(Y, X).fit()
    ## Inspect the results
    print("\n")
    print(mod1res.summary())


if __name__=="__main__":
    filename="./data/Airbnb_Cleaned.csv"
    #data = pd.read_csv(filename,sep=',')
    data=Get_Data(filename)
    
    Boxplot(data)
    category_var_list = {'room_type'}
    print(data['room_type'].unique())
    data=Category_to_Dummy(data,category_var_list)
    
    Corr_HeatMap(data) 
    Linear_model(data)
    

    
    

    
    



