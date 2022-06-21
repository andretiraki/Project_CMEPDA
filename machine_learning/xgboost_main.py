# -*- coding: utf-8 -*-
#pylint: disable=unused-wildcard-import
#pylint: disable=wildcard-import
"""
In this module we use XGBoost Classifier tecnique.
We will try before without optimazing the hyperparameter, than we will do it to improve
our results.
"""
import time
import pandas as pd
from xgboost_tool import *
PATH = r'../data_analysis/data_analyzed.csv'


if __name__ == '__main__':
    start = time.time()
    print('Importing dataset...')
    data = pd.read_csv(PATH)
    print('Dataset imported!')
    #basic without rebalance
    print('Start basic XGBoost without rebalance')
    basic_xgboost(data , rebalance = 0, save = 0)
    #basic with rebalance
    print('Start basic XGBoost with rebalance')
    basic_xgboost(data , rebalance = 1, save = 0)
    #hyper without rebalance
    print('Start hyper XGBoost without rebalance')
    hyper_xgboost(data , rebalance = 0, save = 0)
    #hyper with rebalance
    print('Start hyper XGBoost with rebalance')
    hyper_xgboost(data , rebalance = 1, save = 0)
    mins = (time.time()-start)//60
    sec = (time.time()-start) % 60
    print(f"Elapsed time: {mins} min {sec:.2f} sec\n")
    