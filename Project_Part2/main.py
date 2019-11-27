#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 22:14:18 2018

@author: katezeng

PLEASE READ EVERYONE!!
The main.py file works like a "main" function, the seperated files in this folder are called "Python modules", think of them as libraries that
you can import. To use the functions that defined in modules in our folder, for example, suppose we have "test()" in each module, we call them as:
    DataCleaning.test()
    HistCorr.test()
    ClusterAL.test()
    AssociRule.test()
    PredictAL.test()
"""

#########################################
#    Import Libraries and Modules       #
#########################################
import DataCleaning
import Analysis
import HistCorr
import Proj_clustering
import Associ
import PredictAL

def main():
    #" We'll do sth later ..."
    print("I'm the main function!")


if __name__=="__main__":
    main()