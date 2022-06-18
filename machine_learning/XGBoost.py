# -*- coding: utf-8 -*-
"""
In this module we import function for XGBoost
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_curve, auc, roc_auc_score, precision_recall_curve, confusion_matrix
from imblearn.over_sampling import SMOTE
from collections import Counter
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)



def basic_xgboost(data, rebalance: int = 1, save: int = 0):
    if rebalance != 0 and save != 1: 
        raise ValueError('save must be 1 or 0')
    if save != 0 and save != 1: 
        raise ValueError('save must be 1 or 0')
    print('Preparing train and test...')
    #dividing the dataset in training X and y
    X = data.drop('loan_repaid', axis=1)
    Y = data['loan_repaid']
    #Split data in train and test
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, \
                                                        random_state=100)
    print('Train and test preparation successful!')
    if rebalance == 1 :
        #Using SMOTE to rebalance the unbalance class 'loan repaired'
        print('Rebalancing data...')
        sm = SMOTE(random_state=42)
        X, Y= sm.fit_resample(X, Y)
        print('Resampled dataset shape %s' % Counter(Y))
    print('Starting basic XGBoost...')
    model_xgb = XGBClassifier(seed = 42, early_stopping_rounds = 10, eval_metric ='aucpr')
    xgb = model_xgb.fit(X_train, y_train, verbose =True , eval_set = [(X_test,y_test)])
    #array of predicted class
    y_pred_xgb = model_xgb.predict(X_test)
    print('XG boost done! Printing results:')
    print(f'Accuracy XGB:{metrics.accuracy_score(y_test, y_pred_xgb)}')
    print(f'Precision XGB:{metrics.precision_score(y_test, y_pred_xgb)}')
    print(f'Recall XGB:{metrics.recall_score(y_test, y_pred_xgb)}')
    print(f'F1 Score XGB:{metrics.f1_score(y_test, y_pred_xgb)}')    
    #plotting the confusion matrix    
    plt.figure(figsize=(15,8))
    metrics.plot_confusion_matrix(xgb, X_test, y_test, values_format= 'd', \
                                  display_labels = ['Charged Off', 'Fully Paid'])
    if rebalance == 0:
        out_name='basic_XGBoost.txt'
    else:
        out_name='basic_balanced_XGBoost.txt'
    with open(out_name, "w") as f:
        print(f'Accuracy XGB:{metrics.accuracy_score(y_test, y_pred_xgb)}',file = f)
        print(f'Precision XGB:{metrics.precision_score(y_test, y_pred_xgb)}',file = f)
        print(f'Recall XGB:{metrics.recall_score(y_test, y_pred_xgb)}',file = f)
        print(f'F1 Score XGB:{metrics.f1_score(y_test, y_pred_xgb)}',file = f)   
    if save == 1:
        #Create folder Figure if it does not exist
        if os.path.isdir('Figures') == False:
            print('Creating Folder Figures...\n')
            os.mkdir('Figures')
        script_dir = os.path.dirname(__file__)
        results_dir = os.path.join(script_dir, 'Figures/')
        if rebalance == 0:
            name = '(1)basic_XGBoost.pdf'
        else:
            name= '(2)basic_balanced_XGBoost.pdf'
        plt.savefig(results_dir + name, format="pdf", bbox_inches="tight")


