# -*- coding: utf-8 -*-
"""
The main of this module is printing and save (if you want) plot
"""

import matplotlib.pyplot as plt
import seaborn as sns
import os


def countpl(column , data, save: int = 0, name = 'countplot.pdf', order = None, hue = None):
    '''
    This function plot counts of observations in each categorical bin using bars
    and save it in folder Figures as pdf

    Parameters
    ----------
    column : string
        name of the column of dataframe you want the plot.
    data : DataFrame, array, or list of arrays
        dataframe you want the plot.
    save : int, optional
        it can be 1 (if you want to save) or 0 (if you don't). The default is 0.
    name : string, optional
        name of the picture you want. The default is 'countplot.pdf'.
    order : list of strings, optional
        order the categorical levels
    hue : name of variables in data, optional
        inputs for plotting long-form data, for our aim it will be "loan_status" 
        when we recall the function
    Returns
    -------
    None.

    '''
    #make sure value of save is 0 or 1
    if (type(save) != int or save != 1 and 0): 
        raise ValueError('save must be 1 or 0')
    #Create folder Figure if it does not exist
    if os.path.isdir('Figures') == False:
        print('Creating Folder Figures...\n')
        os.mkdir('Figures')
    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, 'Figures/') 
    plt.figure(figsize=(17,7))
    sns.set_context("paper", font_scale=1)
    sns.countplot(x = column, data = data, palette="seismic", order = order, hue = hue)
    #save figure in Figures folder as pdf
    if save == 1:
        plt.savefig(results_dir + name, format="pdf", bbox_inches="tight")
        

def barpl(column_co, column_fp, save: int = 0, name = 'barplot.pdf'):
    '''
    This function plot the bar plot of the feature that are "Charged Off"
    
    Parameters
    ----------
    column_co : pandas.core.series.Series
        number of elements in column we want that are "Charged Off"
    column_fp : pandas.core.series.Series
        number of elements in column we want that are "Fully Paid".
    save : int, optional
        it can be 1 (if you want to save) or 0 (if you don't). The default is 0.
    name : string, optional
        name of the picture you want. The default is 'barplot.pdf'.

    Returns
    -------
    None.

    '''
    #make sure value of save is 0 or 1
    if (type(save) != int or save != 1 and 0): 
        raise ValueError('save must be 1 or 0') 
    #Create folder Figure if it does not exist
    if os.path.isdir('Figures') == False:
        print('Creating Folder Figures...\n')
        os.mkdir('Figures')
    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, 'Figures/')
    #fraction of feature that are "Charged Off"
    feature_graph = column_co/(column_co + column_fp) 
    plt.figure(figsize=(22,7))
    feature_graph.plot(kind="bar")
    #save figure in Figures folder as pdf
    if save == 1:
        plt.savefig(results_dir + name, format="pdf", bbox_inches="tight")
    
def histpl(column, data, save: int = 0, name = 'histpl.pdf', hue = None, binwidth= None):
        '''
        This function plot counts of observations in each categorical bin using bars
        and save it in folder Figures as pdf

        Parameters
        ----------
        column : string
            name of the column of dataframe you want the plot.
        data : DataFrame, array, or list of arrays
            dataframe you want the plot.
        save : int, optional
            it can be 1 (if you want to save) or 0 (if you don't). The default is 0.
        name : string, optional
            name of the picture you want. The default is 'histpl.pdf'.
        hue : name of variables in data, optional
            inputs for plotting long-form data, for our aim it will be "loan_status" 
            when we recall the function
        binwidth : number or pair of numbeers
                 width of each bin
        Returns
        -------
        None.

        '''
        #make sure value of save is 0 or 1
        if (type(save) != int or save != 1 and 0): 
            raise ValueError('save must be 1 or 0')
        #Create folder Figure if it does not exist
        if os.path.isdir('Figures') == False:
            print('Creating Folder Figures...\n')
            os.mkdir('Figures')
        script_dir = os.path.dirname(__file__)
        results_dir = os.path.join(script_dir, 'Figures/') 
        plt.figure(figsize=(12,7))
        sns.set_context("paper", font_scale=2)
        sns.histplot(x = column, data = data, palette="seismic", hue = hue, binwidth = binwidth)
        #save figure in Figures folder as pdf
        if save == 1:
            plt.savefig(results_dir + name, format="pdf", bbox_inches="tight")

def corr_matrix(data, save: int = 0, name = 'corr_matrix.pdf'):
    '''
    This function compute and plot matrix correlation of our dataset
    Parameters
    ----------
    data : DataFrame, array, or list of arrays+
        dataframe you want to compute matrix correlation.
    save : int, optional
        it can be 1 (if you want to save) or 0 (if you don't). The default is 0.
    name : TYPE, optional
        name of the picture you want. The default is 'corr_matrix.pdf'.

    Returns
    -------
    None.

    '''
    #make sure value of save is 0 or 1
    if (type(save) != int or save != 1 and 0):
        raise ValueError('save must be 1 or 0')
    #Create folder Figure if it does not exist
    if os.path.isdir('Figures') == False:
        print('Creating Folder Figures...\n')
        os.mkdir('Figures')
    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, 'Figures/') 
    plt.figure(figsize=(20,8))
    sns.set_context("paper", font_scale=2)
    sns.heatmap(data.corr(), annot = True, cmap = 'viridis')
    #save figure in Figures folder as pdf
    if save == 1:
        plt.savefig(results_dir + name, format="pdf", bbox_inches="tight")
    return data.corr()
    
    

