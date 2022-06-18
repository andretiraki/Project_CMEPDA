# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 21:12:42 2022

@author: manue
"""
import pandas as pd
from XGBoost import basic_xgboost
import time

if __name__ == '__main__':
    start = time.time()
    print('Importing dataset...')
    data = pd.read_csv('../data_analysis/data_analyzed.csv')
    print('Dataset imported!')
    basic_xgboost(data , rebalance = 1, save = 0)
    mins = (time.time()-start)//60
    sec = (time.time()-start) % 60
    print(f"Elapsed time: {mins} min {sec:.2f} sec\n")
    