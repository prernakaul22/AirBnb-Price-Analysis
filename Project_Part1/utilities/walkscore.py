#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 00:27:46 2018

retrieve data from walkscore API

@author: katezeng
"""

#import pandas as pd
import json
import urllib
from urllib.request import urlopen

def WalkScore():
    baseURL = "http://transit.walkscore.com/transit/score/?"
    api_key = "99947b9fca3cf60a6694dc8f55732300"
    #dfs = list()
    
    # attributes
    latitudes = ['40.7992048', '40.80081594']
    longitudes = ['-73.95367575', '-73.96520163']
    cityname = "New_York"
    statename = "NY"
    
    for latitude, longitude in zip(latitudes, longitudes):
        # example call:
        # http://transit.walkscore.com/transit/score/?lat=47.6101359&lon=-122.3420567&city=Seattle&state=WA&wsapikey=your_key
        scoreURL = baseURL + urllib.parse.urlencode({
                'lat': latitude,
                'lon': longitude,
                'city': cityname,
                'state': statename,
                'wsapikey': api_key 
                })
        print(scoreURL)
    
    response = urlopen(scoreURL).read().decode('utf-8')
    # dump results to a json file
    with open('walkscore.json', 'a') as writeFile:
        json.dump(response, writeFile, sort_keys=True, indent=4)
    
    # for better data handling    
    #df = pd.read_json(response)
    #dfs.append(df)
    
    #results = pd.concat(dfs)
    #columns = ['scored_lat', 'scored_lon', 'transit_score', 'description', 'summary', 'ws_link', 'help_link']
    #return results
    
def main():
    WalkScore()    

if __name__ == "__main__":
    main()