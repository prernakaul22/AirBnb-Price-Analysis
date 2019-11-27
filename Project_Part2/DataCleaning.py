#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 22:14:13 2018

@author: katezeng

This module is for basic statistical analysis and data cleaning insight
    - Determine the mean (mode if categorical), median, and standard deviation of at least 10 attributes in your data sets. 
      Use Python to generate these results and use the project report to show and explain each.
    - In the last assignment you took several steps to clean your data. Here you need to check to make sure that the cleaning decisions 
      you made make sense for the analysis you will do in this assignment. To do this, consider your raw data and consider your current 
      cleaned data. Next, do the following:
          - Identify any attributes that may contain outliers. If you did not deal with outliers in Project 1, do this now by writing 
            Python3 code to locate and potentially clean outliers. In your report, note the attributes that contained potential 
            outliers (you do not have to list all the outliers themselves)
          - Explain how you detected the outliers, and how you made the decision to keep or remove them.
          - From the cleaning phase of Project 1, also discuss which attributes had missing values and explain your strategy for handling them.
          - If you find that you data needs to be further cleaned or differently cleaned based on analyses, include explanations here. Be specific 
            about what you did and why.
    - For at least one of the numeric variables in one of the datasets, write code to bin the data. This will create a new column. 
      Use the binning strategy that is most intuitive for your data. Explain your decision. 
      Include why you chose to bin the specific attribute selected, the binning method used, and why that method makes sense for your data.
"""

import pandas as pd
import numpy as np
import math

# Read file to get wanted variables from the Airbnb data set
def Get_Airbnb_Data(filename):
    data=pd.read_csv(filename,sep=',')
    airbnb_df=pd.DataFrame({'price':data['price'],'property_type':data['property_type'],
                          'room_type':data['room_type'],'accommodates':data['accommodates'],
                          'bathrooms':data['bathrooms'],'bedrooms':data['bedrooms'],
                          'beds':data['beds'],'bed_type':data['bed_type'],
                          'guests_included':data['guests_included'],
                          'host_profile_pic':data['host_has_profile_pic'],'identity_verified':data['host_identity_verified'],
                          'zipcode':data['zipcode'],'latitude':data['latitude'],'longitude':data['longitude']})

        
    # Define lists to store columns' names based on their types
    positive_var_list = {'price','bathrooms','bedrooms','beds','guests_included','zipcode','accommodates','latitude'}
    integer_var_list = {'bedrooms','beds','guests_included','zipcode','accommodates'}    
    bool_var_list = {'host_profile_pic','identity_verified'}
 
    
    # Clean the data
    airbnb_df=Data_Cleaning(airbnb_df,positive_var_list,integer_var_list)
    
    return (airbnb_df)


# Transform Categorical variables into dummies:     
def Category_to_Dummy(data,category_var_list):
    for i in category_var_list:
        # Create dummies variables and concatenate them
        dummies=pd.get_dummies(data[i])
        data=pd.concat([data,dummies],axis=1)
        
        # Drop the categorical variable
        data=data.drop([i],axis=1)
        
    return (data)


# Clean the data
def Data_Cleaning(data,positive_var_list=None,integer_var_list=None):
    data=data.dropna() 
    
    # Drop variables is not positive
    if positive_var_list!=None:
        for i in positive_var_list:
            data=data.loc[~(data[i]<0)] 
     
    # Drop variables is not integer
    if integer_var_list!=None:
        for i in integer_var_list:
            data=data.loc[~(data[i].apply(lambda x: math.floor(x)!= x))]
    
    # Standarized the zip code
    data['zipcode']=data['zipcode'].apply(str)
    for i in data['zipcode']:
        # First remove all space
        i = i.replace(" ", "")
        # Only select first 5 number
        if len(i)>5:
            i = i[0:5]
    
    data = data[data['zipcode'].apply(lambda x:len(x)==5)]
    return (data)
    
    
# Change upper case into lower case  
def LowerCase(data,var):
    for i in var:
        data[i]=data[i].str.lower()
    return (data)


if __name__ == "__main__":
    # Generate new features and check Airbnb Cleanliness
    airbnb_df=Get_Airbnb_Data(filename="./data/Airbnb_listings.csv")
    

    # Merge three data sets 
    hotel_df = pd.read_csv("./data/hotel_data.csv",sep=',')
    hotel_df['zipcode']=hotel_df['zipcode'].apply(str)
    yelp_df = pd.read_csv("./data/Yelp_num_res.csv",sep=',')
    yelp_df['zipcode']=yelp_df['zipcode'].apply(str)
    airbnb_df=pd.merge(airbnb_df,hotel_df, on='zipcode', how='left')
    airbnb_df=pd.merge(airbnb_df,yelp_df, on='zipcode', how='left')
    airbnb_df=airbnb_df.dropna() 
    airbnb_df=airbnb_df[(0<airbnb_df['price'])&(airbnb_df['price']<=1000)]
    airbnb_df.to_csv('./data/Airbnb_Cleaned.csv', sep=',', encoding='utf-8')
    #with open('Airbnb_Cleaned.csv','w') as f:
   





















































