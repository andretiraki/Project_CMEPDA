# -*- coding: utf-8 -*-
#pylint: disable=invalid-name
#pylint: disable=too-many-locals
#pylint: disable=consider-using-f-string
#pylint: disable=line-too-long
"""
In this module we import function for XGBoost
"""
import os
import warnings
from collections import Counter
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split,GridSearchCV
from imblearn.over_sampling import SMOTE
warnings.simplefilter(action='ignore', category=FutureWarning)

SEED = 42

def basic_xgboost(data, rebalance: int = 1, save: int = 0):
    '''
    This is basic form of XG Boost. If you want to rebalance the data: 'rebalance' = 1
    if you don't ''rebalance' = 0

    Parameters
    ----------
    data : DataFrame
        dataframe you want to use the tecnique.
    rebalance : int, optional
        it can be 1 (if you wanto to rebalance your data) or 0 (if you don't).
        The default is 1.
    save : int, optional
        it can be 1 (if you want to save) or 0 (if you don't). The default is 0.

    Raises
    ------
    ValueError
        rebalance and save must be 0 or 1.

    Returns
    -------
    None.

    '''
    if save not in (0,1):
        raise ValueError('save must be 1 or 0')
    if rebalance not in (0,1):
        raise ValueError('rebalance must be 1 or 0')
    print('Preparing train and test...')
    #dividing the dataset in training X and Y
    X = data.drop('loan_repaid', axis=1)
    Y = data['loan_repaid']
    #Split data in train and test
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, \
                                                        random_state=100)
    print('Train and test preparation successful!')
    if rebalance == 1 :
        #Using SMOTE to rebalance the unbalance class 'loan repaired'
        print('Rebalancing data...')
        sm = SMOTE(random_state = SEED)
        X, Y= sm.fit_resample(X, Y)
        print('Resampled dataset shape %s' % Counter(Y))
    print('Starting basic XGBoost...')
    model_xgb = XGBClassifier(seed = SEED, early_stopping_rounds = 10, eval_metric ='aucpr')
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
    with open(out_name, 'w',encoding='utf-8') as file:
        print(f'Accuracy XGB:{metrics.accuracy_score(y_test, y_pred_xgb)}',file = file)
        print(f'Precision XGB:{metrics.precision_score(y_test, y_pred_xgb)}',file = file)
        print(f'Recall XGB:{metrics.recall_score(y_test, y_pred_xgb)}',file = file)
        print(f'F1 Score XGB:{metrics.f1_score(y_test, y_pred_xgb)}',file = file)
    if save == 1:
        #Create folder Figure if it does not exist
        if os.path.isdir('Figures') is False:
            print('Creating Folder Figures...\n')
            os.mkdir('Figures')
        script_dir = os.path.dirname(__file__)
        results_dir = os.path.join(script_dir, 'Figures/')
        if rebalance == 0:
            name = '(1)basic_XGBoost.pdf'
        else:
            name= '(2)basic_balanced_XGBoost.pdf'
        plt.savefig(results_dir + name, format="pdf", bbox_inches="tight")


def hyper_xgboost(data, rebalance: int = 1, save: int = 0):
    '''
    This is hyperparameter form of XG Boost. If you want to rebalance the data: 'rebalance' = 1
    if you don't ''rebalance' = 0

    Parameters
    ----------
    data : DataFrame
        dataframe you want to use the tecnique.
    rebalance : int, optional
        it can be 1 (if you wanto to rebalance your data) or 0 (if you don't).
        The default is 1.
    save : int, optional
        it can be 1 (if you want to save) or 0 (if you don't). The default is 0.

    Raises
    ------
    ValueError
        rebalance and save must be 0 or 1.

    Returns
    -------
    None.

    '''
    if save not in (0,1):
        raise ValueError('save must be 1 or 0')
    if rebalance not in (0,1):
        raise ValueError('rebalance must be 1 or 0')
    print('Preparing train and test...')
    #dividing the dataset in training X and Y
    X = data.drop('loan_repaid', axis=1)
    Y = data['loan_repaid']
    #Split data in train and test
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, \
                                                        random_state=100)
    print('Train and test preparation successful!')
    if rebalance == 1 :
        #Using SMOTE to rebalance the unbalance class 'loan repaired'
        print('Rebalancing data...')
        sm = SMOTE(random_state = SEED)
        X, Y= sm.fit_resample(X, Y)
        print('Resampled dataset shape %s' % Counter(Y))
    #Optimaze the hyperparameter
    ## ROUND 1
    param_grid = {
        'max_depth': [3, 4, 5],
        'learning_rate': [0.1, 0.01, 0.05],
        'reg_lambda': [0, 1.0, 10.0],
        'early_stopping_rounds': [10],
        'eval_metric' : ['auc'],
    }
    #ROUND2
    param_grid = {
        'max_depth':[4],
        'learning_rate': [0.1, 0.5, 1],
        'gamma': [0.25],
        'reg_lambda':[10.0,20, 100],
        'early_stopping_rounds': [10],
        'eval_metric' : ['auc']
    }
    #Searching the best parameters
    optimal_params = GridSearchCV(estimator=XGBClassifier(seed = 42, subsamaple=0.9, colsample_bytree= 0.5),
                                  param_grid = param_grid,
                                  scoring='roc_auc',
                                  verbose=2,
                                  cv = 5
                                 )
    optimal_params.fit(X_train,
                       y_train,
                       eval_set=[(X_test, y_test)],
                       verbose=False)
    print(optimal_params.best_params_)

    #Construct the XGBoost classifier based on the best hyperparameters found
    model_xgb_hpt = XGBClassifier(seed = 42,
                                  gamma = optimal_params.best_params_['gamma'],
                                  learning_rate = optimal_params.best_params_['learning_rate'],
                                  max_depth = optimal_params.best_params_['max_depth'],
                                  reg_lambda = optimal_params.best_params_['reg_lambda'],
                                  early_stopping_rounds = optimal_params.best_params_['early_stopping_rounds'],
                                  eval_metric = optimal_params.best_params_['eval_metric'])
    xgb_hpt = model_xgb_hpt.fit(X_train, y_train, verbose =True , eval_set = [(X_test,y_test)])
    y_pred_xgb = model_xgb_hpt.predict(X_test)
    #plotting confusion matrix
    metrics.plot_confusion_matrix(xgb_hpt, X_test, y_test, values_format= 'd', \
                                  display_labels = ['Charged Off', 'Fully Paid'])
    if rebalance == 0:
        out_name='hyper_XGBoost.txt'
    else:
        out_name='hyper_balanced_XGBoost.txt'
    with open(out_name, 'w',encoding='utf-8') as file:
        print(f'Accuracy XGB:{metrics.accuracy_score(y_test, y_pred_xgb)}',file = file)
        print(f'Precision XGB:{metrics.precision_score(y_test, y_pred_xgb)}',file = file)
        print(f'Recall XGB:{metrics.recall_score(y_test, y_pred_xgb)}',file = file)
        print(f'F1 Score XGB:{metrics.f1_score(y_test, y_pred_xgb)}',file = file)
    if save == 1:
        #Create folder Figure if it does not exist
        if os.path.isdir('Figures') is False:
            print('Creating Folder Figures...\n')
            os.mkdir('Figures')
        script_dir = os.path.dirname(__file__)
        results_dir = os.path.join(script_dir, 'Figures/')
        if rebalance == 0:
            name = '(3)hyper_XGBoost.pdf'
        else:
            name= '(4)hyper_balanced_XGBoost.pdf'
        plt.savefig(results_dir + name, format="pdf", bbox_inches="tight")
