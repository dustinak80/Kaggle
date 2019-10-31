# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 20:57:30 2019

@author: Dustin
"""
import pandas as pd
from matplotlib import pyplot as plt 
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from scipy import stats


file_training = r'C:\Users\Dustin\Desktop\Machine Learning\kaggle\house-prices-advanced-regression-techniques\train.csv'
#import the training Data
training = pd.read_csv(file_training) 
training.index = training['Id']
training.drop('Id', axis = 1, inplace = True)

#Seperate the features between categorical and numerical
def features_to_dic(data):
    l = data.columns.tolist()
    dic = {}
    for i in range(len(l)):
        dic[l[i]] = i
    
    return dic

features_dic = features_to_dic(training)

### FIND NULL VALUES

#change year built
training.GarageYrBlt[training['GarageYrBlt'].isnull()] = \
training.YearBuilt[training['GarageYrBlt'].isnull()]


### CONVERT OBJECTS TO DUMMIE VARIABLES
test1 = pd.get_dummies(training)

### LOOK AT CORRELATION
#Correlation matrix
corr_matrix = test1.corr()
#Look at values with a high correlation to Sales Price
med_corr = corr_matrix[abs(corr_matrix['SalePrice']) > .10]
med_corr_list = list(med_corr.index)

#make a dataframe with the high correlation features
training_mc = test1[med_corr_list]

### DO SOME DATA PRE-PROCESSING

#(TotalBsmtSF, 1stFlrSF) and (GarageCars, GarageArea) have high correlation with eachother
training_mc.drop(['1stFlrSF', 'GarageArea'], axis = 1, inplace = True)

#look at null values
train.isnull().sum()
def find_null(data):
    """
    Searches through each column to find the index placement of null values
    data: data to be searched
    """
    columns = data.columns.tolist()
    dic_nulls = {}
    for i in range(len(columns)):
        #some dont have any NaN's, so do try loop so dont get code
        try:
            check = sum(data[columns[i]].isnull())
        except KeyError:
            pass
        if check > 0:
            #create tuple for column name - (list of row index for null values, index of column placement)
            dic_nulls[columns[i]] = (data[data[columns[i]].isnull()].index.values,i)
        else:
            dic_nulls[columns[i]] = (0,i)
    return dic_nulls

mc_null = find_null(training_mc)

### REPLACE NULL VALUES
#For lot Frontage
X_limit = training_mc[training_mc['LotArea'] < 25000] #Take out outliers for LotArea
X_limit = X_limit[X_limit['LotFrontage'] < 125] #Take out outliers for LotArea
slope, intercept, r_value, p_value, std_err = \
    stats.linregress(X_limit.LotArea[X_limit['LotFrontage'].notnull()], \
                                     X_limit.LotFrontage[X_limit['LotFrontage'].notnull()])
training_mc.loc[mc_null['LotFrontage'][0],'LotFrontage'] = \
    round(training_mc.loc[mc_null['LotFrontage'][0],'LotArea']*slope+intercept)
#MAsVnrArea (null = 0), GarageYearBuilt (null = 0)
training_mc[['MasVnrArea', 'GarageYrBlt']].fillna(0, inplace = True)

check = find_null(training_mc)

### RANDOM FORREST REGRESSION
#split into X, y
y_mc = training_mc['SalePrice']
X_mc = training_mc.drop('SalePrice', axis = 1)

def rand_forrest_regression(X, y, n_est = [25,75], samples_split = [5,10], \
                            predictions = None, feature_importance = None, \
                            best_score = None ):
    """
    X, y = features dataframe and response variable
    n_est = range for # of estimators (list format)
    samples_split = range for # of sample to split the data
    predictions = in case other runs have been ran bring in dictionary with values
    """
    fsz = (12,8)
    
    #run the loop that goes through # of trees
    n_est_score = {}
    for i in n_est:
        regr = RandomForestRegressor(random_state=0,
                                     n_estimators=i, min_samples_split = 10)
        regr.fit(X, y)
        n_est_score[regr.score(X, y)] = i

    #run the loop that goes through min # of samples for a split
    sam_split_score = {}
    for i in samples_split:
        regr = RandomForestRegressor(random_state=0,
                                     n_estimators=n_est_score[max(n_est_score)], min_samples_split = i)
        regr.fit(X, y)
        sam_split_score[regr.score(X,y)] = i
    
    #Get predictions and feature importance for the best score of the two loops
    regr = RandomForestRegressor(random_state=0,
                                     n_estimators=n_est_score[max(n_est_score)], 
                                     min_samples_split = sam_split_score[max(sam_split_score)])
    regr.fit(X,y)
    if feature_importance == None:
        feature_importance = {}
    if predictions == None:
        predictions = {}
    if best_score == None:
        best_score = {}
    feature_importance[(n_est_score[max(n_est_score)],sam_split_score[max(sam_split_score)])] = regr.feature_importances_
    predictions[(n_est_score[max(n_est_score)],sam_split_score[max(sam_split_score)])] = regr.predict(X)
    best_score[max(sam_split_score)] = (n_est_score[max(n_est_score)],sam_split_score[max(sam_split_score)])

    #plot the scores for multiple n_estimators (# of trees) and multiple min samples   
    fig, ax= plt.subplots(figsize = fsz)
    ax.scatter(n_est_score.values(), n_est_score.keys())
    ax.plot(list(n_est_score.values()), list(n_est_score.keys()), c = 'r')
    ax.set_title('Score vs. Number of Trees in Forest')
    ax.set_ylabel('Score')
    ax.set_xlabel('Number of Trees in Forest')
    
    fig2, ax2 = plt.subplots(figsize = fsz)
    ax2.scatter(sam_split_score.values(), sam_split_score.keys())
    ax2.plot(list(sam_split_score.values()), list(sam_split_score.keys()))
    ax2.set_title('Score vs. Minimum Number of Samples in a Leaf')
    ax2.set_ylabel('Score')
    ax2.set_xlabel('Minimum Number of Samples in a Leaf ({} trees)'.format(n_est_score[max(n_est_score)]))
    
    return predictions, feature_importance, best_score

predictions, feature_importance, best_score = \
    rand_forrest_regression(X_mc, y_mc,n_est = [10, 25, 40, 50, 60, 75, 100],
                            samples_split = [30, 40, 50, 60, 70])

#FOUND A GOOD SPOT TO BE 40


play = training.isnull().sum()
play_vals = play[play.values > 0]

