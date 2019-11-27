#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 16:44:35 2018

@author: stevechen
"""
import plotly
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go
import pandas as pd
import plotly.figure_factory as ff
import numpy as np

# The function is to draw a picture to show rows and column comparison 
# before and after cleaning and megring
def Com_Data (data1,data2,title,filename):
    
    trace1 = go.Scatter(x=[0,1], y=data1,name='Number of Rows')
    trace2 = go.Scatter(x=[1,2], y=data2,name='Number of Columns',yaxis='y2')
    data = [trace1, trace2]
    layout = go.Layout(title=title,yaxis=dict(title='Number of Rows'),
                       yaxis2=dict(title='Number of Rows',titlefont=dict(
                        color='rgb(148, 103, 189)'),
    tickfont=dict(color='rgb(148, 103, 189)'),overlaying='y',side='right'))
    
    fig = go.Figure(data=data, layout=layout)
    #py.plot(fig, filename=title)
    plotly.offline.plot(fig, filename=filename)

# A pie chart to show proportion about three data sets both in rows and columns
def Pie_Chart(data1,data2,title,filename):
    fig = {"data": [{"values":data1,
      "labels": [
        "AirBnb",
        "Hotel",
        "Yelp"],
      "domain": {"x": [0, .48]},
      "name": "Number of Rows",
      "hoverinfo":"label+value+name",
      "hole": .4,
      "type": "pie"
    },
    {
      "values":data2, "labels": ["AirBnb","Hotel","Yelp"],
      "textposition":"inside","domain": {"x": [.52, 1]},
      "name": "Number of Columns","hoverinfo":"label+value+name","hole": .4,
      "type": "pie"}],
    "layout": {
        "title":title,
        "annotations": [{"font": {"size": 20},"showarrow": False,"text": "Rows","x": 0.20,
                "y": 0.5},{"font": {"size": 20},"showarrow": False,"text": "Columns","x": 0.82,"y": 0.5}]}}
    
    #py.plot(fig, filename=title)
    plotly.offline.plot(fig, filename=filename)

# Violin chart to compare price among different room types
def Violin_Chart(data):
    data_private = data[data['room_type']=='Private room']
    data_entire = data[data['room_type']=='Entire home/apt']
    data_shared = data[data['room_type']=='Shared room']
    
    trace1 = {"type": 'violin',"y":data_private['price'],"name":'Private room',
              "box": {"visible": True},"meanline": {"visible": True},
              "opacity": 0.6,"x0": "Private room"}
    
    trace2 = {"type": 'violin',"y":data_entire['price'],"name":"Entire home/apt",
              "box": {"visible": True},"meanline": {"visible": True},
              "opacity": 0.6,"x0": "Entire home/apt"}
    
    trace3 = {"type": 'violin',"y":data_shared['price'],"name":"Shared room",
              "box": {"visible": True},"meanline": {"visible": True},
            "opacity": 0.6,"x0": "Shared room"}
        
    
    fig1 = {"data": [trace1,trace2,trace3],"layout" : {"title": "Violin Plot for Airbnb Price",
                                          "yaxis": {"zeroline": False}}}
    
    py.iplot(fig1, filename='violin_price')
    plotly.offline.plot(fig1, filename='Violin_Price.html')
    #data1 = data.iloc[:10000,:]
    data1 =data
    data_private = data1[data1['room_type']=='Private room']
    data_entire = data1[data1['room_type']=='Entire home/apt']
    data_shared = data1[data1['room_type']=='Shared room']
    trace4 = dict(name ='Private room',
                  x=data_private['accommodates'],y=data_private['guests_included'],
                          z=data_private['price'],type = "scatter3d",
                          mode='markers',marker=dict(size=3, color='rgb(228,26,28)',     # set color to an array/list of desired values
                        line=dict(width=0)))
    
    trace5 = dict(name ='Entire home/apt',
                  x=data_entire['accommodates'],y=data_entire['guests_included'],
                          z=data_entire['price'],type = "scatter3d",
                          mode='markers',marker=dict(size=3, color='rgb(55,126,184)',     # set color to an array/list of desired values
                        line=dict(width=0)))

    trace6 = dict(name ='Shared room',
                  x=data_shared['accommodates'],y=data_shared['guests_included'],
                          z=data_shared['price'],type = "scatter3d",
                          mode='markers',marker=dict(size=3, color='rgb(77,175,74)',     # set color to an array/list of desired values
                        line=dict(width=0)))   
    
    
    data_tra = [trace4,trace5,trace6]
            
    layout = dict(width=1000,height=1000,autosize=False,
                  title='Price v.s. Accomodates v.s. Guests_included',
                 scene=dict(
                xaxis=dict(title='accommodates',gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',showbackground=True,backgroundcolor='rgb(230, 230,230)'),
                yaxis=dict(title='guests_included',gridcolor='rgb(255, 255, 255)',
            zerolinecolor='rgb(255, 255, 255)',showbackground=True,backgroundcolor='rgb(230, 230,230)'),
            zaxis=dict(title='price',gridcolor='rgb(255, 255, 255)',zerolinecolor='rgb(255, 255, 255)',
            showbackground=True,backgroundcolor='rgb(230, 230,230)'), 
            aspectratio = dict( x=1, y=1, z=0.7 ),aspectmode = 'manual'))
    
    
    fig = dict(data=data_tra, layout=layout)
    py.iplot(fig, filename='3D_Scatters')
    plotly.offline.plot(fig, filename='3D_Scatters.html')
    
def main():
    airb = pd.read_csv("Airbnb_listings.csv")
    airb_cleaned = pd.read_csv("Airbnb_Cleaned.csv")
    hotel = pd.read_csv("Hotel_prices.csv")
    yelp = pd.read_csv("YelpData.csv")
    num_rows_b = airb.shape[0]+hotel.shape[0]+yelp.shape[0]
    num_cols_b = airb.shape[1]+hotel.shape[1]+yelp.shape[1]
    data1 = [num_rows_b,airb_cleaned.shape[0]]
    data2 = [num_cols_b,airb_cleaned.shape[1]]
    
    
    title1 = 'Data Dimensionality before and after cleaned&merged'
    filename1 = 'data_dim.html'
    #Com_Data(data1,data2,title1,filename1)
    
    # Pie Chart
    data3 = [airb.shape[0],hotel.shape[0],yelp.shape[0]]
    data4 = [airb.shape[1],hotel.shape[1],yelp.shape[1]]
    title2 = 'Pie Chart for Raw Data'
    filename2 = 'PieChart.html'
    #Pie_Chart(data3,data4,title2,filename2)
    
    # 
    #pd.set_option('display.max_columns', 20)
    #print(airb_cleaned['room_type'].unique())
    # Violin Chart for price among various room types
    Violin_Chart(airb_cleaned)
    #Three_D(data)
    # 
    #print(airb_cleaned['guests_included'].unique())
    
if __name__ == '__main__':
    main()
