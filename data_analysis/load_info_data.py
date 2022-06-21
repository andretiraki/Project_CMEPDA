# -*- coding: utf-8 -*-
"""
The main of this module is loading our datafrane and print its info
"""
import time
import pandas as pd



def load_info_data(data_path):
    '''
    This function read a dataframe csv from path and print its info
    ----------
    data_path : String
        dataframe which we want info .

    Returns
    -------
    data: Dataframe

    '''
    start = time.time()
    data = pd.read_csv(data_path)
    print('Loading data...\n')
    mins = (time.time()-start)//60
    sec = (time.time()-start) % 60
    data.info()
    print(f"Time to load the dataset is: {mins} min {sec:.2f} sec\n")
    return data
