#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 16:07:17 2018

@author: stevechen
"""

import pandas as pd
import urllib
from urllib.request import urlopen
import json

def UseUrllib(BaseURL,URLPost):
    # Combine information into a URL
    URL=BaseURL + "?"+ urllib.parse.urlencode(URLPost) 
    
    # Open URL
    WebURL=urlopen(URL)
    
    # Read the URL
    data=WebURL.read()
    
    # Encoding
    encoding = WebURL.info().get_content_charset('utf-8')
    
    # Store the data and return the data
    jsontxt = json.loads(data.decode(encoding))         
    return jsontxt

def main():
    lat=40.7992048
    lon=-73.95367575
    BaseURL="https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    URLPost={'location': str(lat)+','+str(lon),
             'radius':500,
             'type':'subway_station',
             'key':'AIzaSyBfa7ZFgSUsx3JTGcvM_SGq7jcHqay7fpI'}
    jsontxt=UseUrllib(BaseURL,URLPost)
    with open('google_data.json', 'w') as f:
        json.dump(jsontxt, f)
    f.close()

if __name__ == "__main__":
    main()

