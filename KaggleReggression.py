# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 16:37:36 2019

@author: ryanl
"""

# Kaggle Regression

# Import Libraries

import numpy as np
import pandas as pd

# Import Data

train = pd.read_csv("train.csv")

# Descriptive Stats
train.describe()
#                Id   MSSubClass  ...       YrSold      SalePrice
#count  1460.000000  1460.000000  ...  1460.000000    1460.000000
#mean    730.500000    56.897260  ...  2007.815753  180921.195890
#std     421.610009    42.300571  ...     1.328095   79442.502883
#min       1.000000    20.000000  ...  2006.000000   34900.000000
#25%     365.750000    20.000000  ...  2007.000000  129975.000000
#50%     730.500000    50.000000  ...  2008.000000  163000.000000
#75%    1095.250000    70.000000  ...  2009.000000  214000.000000
#max    1460.000000   190.000000  ...  2010.000000  755000.000000

# Features List

train_columns = list(train.columns)

###############################################################################

#['Id',
# 'MSSubClass',
# 'LotFrontage',
# 'LotArea',
# 'OverallQual',
# 'OverallCond',
# 'YearBuilt',
# 'YearRemodAdd',
# 'MasVnrArea',
# 'BsmtFinSF1',
# 'BsmtFinSF2',
# 'BsmtUnfSF',
# 'TotalBsmtSF',
# '1stFlrSF',
# '2ndFlrSF',
# 'LowQualFinSF',
# 'GrLivArea',
# 'BsmtFullBath',
# 'BsmtHalfBath',
# 'FullBath',
# 'HalfBath',
# 'BedroomAbvGr',
# 'KitchenAbvGr',
# 'TotRmsAbvGrd',
# 'Fireplaces',
# 'GarageYrBlt',
# 'GarageCars',
# 'GarageArea',
# 'WoodDeckSF',
# 'OpenPorchSF',
# 'EnclosedPorch',
# '3SsnPorch',
# 'ScreenPorch',
# 'PoolArea',
# 'MiscVal',
# 'MoSold',
# 'YrSold',
# 'SalePrice',
# 'MSZoning_C (all)',
# 'MSZoning_FV',
# 'MSZoning_RH',
# 'MSZoning_RL',
# 'MSZoning_RM',
# 'Street_Grvl',
# 'Street_Pave',
# 'Alley_Grvl',
# 'Alley_Pave',
# 'LotShape_IR1',
# 'LotShape_IR2',
# 'LotShape_IR3',
# 'LotShape_Reg',
# 'LandContour_Bnk',
# 'LandContour_HLS',
# 'LandContour_Low',
# 'LandContour_Lvl',
# 'Utilities_AllPub',
# 'Utilities_NoSeWa',
# 'LotConfig_Corner',
# 'LotConfig_CulDSac',
# 'LotConfig_FR2',
# 'LotConfig_FR3',
# 'LotConfig_Inside',
# 'LandSlope_Gtl',
# 'LandSlope_Mod',
# 'LandSlope_Sev',
# 'Neighborhood_Blmngtn',
# 'Neighborhood_Blueste',
# 'Neighborhood_BrDale',
# 'Neighborhood_BrkSide',
# 'Neighborhood_ClearCr',
# 'Neighborhood_CollgCr',
# 'Neighborhood_Crawfor',
# 'Neighborhood_Edwards',
# 'Neighborhood_Gilbert',
# 'Neighborhood_IDOTRR',
# 'Neighborhood_MeadowV',
# 'Neighborhood_Mitchel',
# 'Neighborhood_NAmes',
# 'Neighborhood_NPkVill',
# 'Neighborhood_NWAmes',
# 'Neighborhood_NoRidge',
# 'Neighborhood_NridgHt',
# 'Neighborhood_OldTown',
# 'Neighborhood_SWISU',
# 'Neighborhood_Sawyer',
# 'Neighborhood_SawyerW',
# 'Neighborhood_Somerst',
# 'Neighborhood_StoneBr',
# 'Neighborhood_Timber',
# 'Neighborhood_Veenker',
# 'Condition1_Artery',
# 'Condition1_Feedr',
# 'Condition1_Norm',
# 'Condition1_PosA',
# 'Condition1_PosN',
# 'Condition1_RRAe',
# 'Condition1_RRAn',
# 'Condition1_RRNe',
# 'Condition1_RRNn',
# 'Condition2_Artery',
# 'Condition2_Feedr',
# 'Condition2_Norm',
# 'Condition2_PosA',
# 'Condition2_PosN',
# 'Condition2_RRAe',
# 'Condition2_RRAn',
# 'Condition2_RRNn',
# 'BldgType_1Fam',
# 'BldgType_2fmCon',
# 'BldgType_Duplex',
# 'BldgType_Twnhs',
# 'BldgType_TwnhsE',
# 'HouseStyle_1.5Fin',
# 'HouseStyle_1.5Unf',
# 'HouseStyle_1Story',
# 'HouseStyle_2.5Fin',
# 'HouseStyle_2.5Unf',
# 'HouseStyle_2Story',
# 'HouseStyle_SFoyer',
# 'HouseStyle_SLvl',
# 'RoofStyle_Flat',
# 'RoofStyle_Gable',
# 'RoofStyle_Gambrel',
# 'RoofStyle_Hip',
# 'RoofStyle_Mansard',
# 'RoofStyle_Shed',
# 'RoofMatl_ClyTile',
# 'RoofMatl_CompShg',
# 'RoofMatl_Membran',
# 'RoofMatl_Metal',
# 'RoofMatl_Roll',
# 'RoofMatl_Tar&Grv',
# 'RoofMatl_WdShake',
# 'RoofMatl_WdShngl',
# 'Exterior1st_AsbShng',
# 'Exterior1st_AsphShn',
# 'Exterior1st_BrkComm',
# 'Exterior1st_BrkFace',
# 'Exterior1st_CBlock',
# 'Exterior1st_CemntBd',
# 'Exterior1st_HdBoard',
# 'Exterior1st_ImStucc',
# 'Exterior1st_MetalSd',
# 'Exterior1st_Plywood',
# 'Exterior1st_Stone',
# 'Exterior1st_Stucco',
# 'Exterior1st_VinylSd',
# 'Exterior1st_Wd Sdng',
# 'Exterior1st_WdShing',
# 'Exterior2nd_AsbShng',
# 'Exterior2nd_AsphShn',
# 'Exterior2nd_Brk Cmn',
# 'Exterior2nd_BrkFace',
# 'Exterior2nd_CBlock',
# 'Exterior2nd_CmentBd',
# 'Exterior2nd_HdBoard',
# 'Exterior2nd_ImStucc',
# 'Exterior2nd_MetalSd',
# 'Exterior2nd_Other',
# 'Exterior2nd_Plywood',
# 'Exterior2nd_Stone',
# 'Exterior2nd_Stucco',
# 'Exterior2nd_VinylSd',
# 'Exterior2nd_Wd Sdng',
# 'Exterior2nd_Wd Shng',
# 'MasVnrType_BrkCmn',
# 'MasVnrType_BrkFace',
# 'MasVnrType_None',
# 'MasVnrType_Stone',
# 'ExterQual_Ex',
# 'ExterQual_Fa',
# 'ExterQual_Gd',
# 'ExterQual_TA',
# 'ExterCond_Ex',
# 'ExterCond_Fa',
# 'ExterCond_Gd',
# 'ExterCond_Po',
# 'ExterCond_TA',
# 'Foundation_BrkTil',
# 'Foundation_CBlock',
# 'Foundation_PConc',
# 'Foundation_Slab',
# 'Foundation_Stone',
# 'Foundation_Wood',
# 'BsmtQual_Ex',
# 'BsmtQual_Fa',
# 'BsmtQual_Gd',
# 'BsmtQual_TA',
# 'BsmtCond_Fa',
# 'BsmtCond_Gd',
# 'BsmtCond_Po',
# 'BsmtCond_TA',
# 'BsmtExposure_Av',
# 'BsmtExposure_Gd',
# 'BsmtExposure_Mn',
# 'BsmtExposure_No',
# 'BsmtFinType1_ALQ',
# 'BsmtFinType1_BLQ',
# 'BsmtFinType1_GLQ',
# 'BsmtFinType1_LwQ',
# 'BsmtFinType1_Rec',
# 'BsmtFinType1_Unf',
# 'BsmtFinType2_ALQ',
# 'BsmtFinType2_BLQ',
# 'BsmtFinType2_GLQ',
# 'BsmtFinType2_LwQ',
# 'BsmtFinType2_Rec',
# 'BsmtFinType2_Unf',
# 'Heating_Floor',
# 'Heating_GasA',
# 'Heating_GasW',
# 'Heating_Grav',
# 'Heating_OthW',
# 'Heating_Wall',
# 'HeatingQC_Ex',
# 'HeatingQC_Fa',
# 'HeatingQC_Gd',
# 'HeatingQC_Po',
# 'HeatingQC_TA',
# 'CentralAir_N',
# 'CentralAir_Y',
# 'Electrical_FuseA',
# 'Electrical_FuseF',
# 'Electrical_FuseP',
# 'Electrical_Mix',
# 'Electrical_SBrkr',
# 'KitchenQual_Ex',
# 'KitchenQual_Fa',
# 'KitchenQual_Gd',
# 'KitchenQual_TA',
# 'Functional_Maj1',
# 'Functional_Maj2',
# 'Functional_Min1',
# 'Functional_Min2',
# 'Functional_Mod',
# 'Functional_Sev',
# 'Functional_Typ',
# 'FireplaceQu_Ex',
# 'FireplaceQu_Fa',
# 'FireplaceQu_Gd',
# 'FireplaceQu_Po',
# 'FireplaceQu_TA',
# 'GarageType_2Types',
# 'GarageType_Attchd',
# 'GarageType_Basment',
# 'GarageType_BuiltIn',
# 'GarageType_CarPort',
# 'GarageType_Detchd',
# 'GarageFinish_Fin',
# 'GarageFinish_RFn',
# 'GarageFinish_Unf',
# 'GarageQual_Ex',
# 'GarageQual_Fa',
# 'GarageQual_Gd',
# 'GarageQual_Po',
# 'GarageQual_TA',
# 'GarageCond_Ex',
# 'GarageCond_Fa',
# 'GarageCond_Gd',
# 'GarageCond_Po',
# 'GarageCond_TA',
# 'PavedDrive_N',
# 'PavedDrive_P',
# 'PavedDrive_Y',
# 'PoolQC_Ex',
# 'PoolQC_Fa',
# 'PoolQC_Gd',
# 'Fence_GdPrv',
# 'Fence_GdWo',
# 'Fence_MnPrv',
# 'Fence_MnWw',
# 'MiscFeature_Gar2',
# 'MiscFeature_Othr',
# 'MiscFeature_Shed',
# 'MiscFeature_TenC',
# 'SaleType_COD',
# 'SaleType_CWD',
# 'SaleType_Con',
# 'SaleType_ConLD',
# 'SaleType_ConLI',
# 'SaleType_ConLw',
# 'SaleType_New',
# 'SaleType_Oth',
# 'SaleType_WD',
# 'SaleCondition_Abnorml',
# 'SaleCondition_AdjLand',
# 'SaleCondition_Alloca',
# 'SaleCondition_Family',
# 'SaleCondition_Normal',
# 'SaleCondition_Partial']

###############################################################################

# Find NA's 

numna = train.isnull().sum()
numna_col = numna[numna.values > 0]
#LotFrontage      259
#Alley           1369
#MasVnrType         8
#MasVnrArea         8
#BsmtQual          37
#BsmtCond          37
#BsmtExposure      38
#BsmtFinType1      37
#BsmtFinType2      38
#Electrical         1
#FireplaceQu      690
#GarageType        81
#GarageYrBlt       81
#GarageFinish      81
#GarageQual        81
#GarageCond        81
#PoolQC          1453
#Fence           1179
#MiscFeature     1406

# Fill Na's
# We are going to fill LotFrontage with 0; we assume N/A means no lot frontage
train['LotFrontage'].fillna(0, inplace = True)

# We are going to fill Alley with None because we are assuming no alley access
train['Alley'].fillna('None', inplace = True)

# We are going to fill MasVnrType with 0 because we are assuming no brick
train['MasVnrType'].fillna(0, inplace = True)

# We are going to fill MasVnrArea with 0 because we are assuming no brick
train['MasVnrArea'].fillna(0, inplace = True)

# We are going to fill BsmtQual with none because we are assuming no basement
train['BsmtQual'].fillna('None', inplace = True)

# We are going to fill BsmtCond with none because we are assuming no basement
train['BsmtCond'].fillna('None', inplace = True)

# We are going to fill BsmtExposure with none because we are assuming no basement
train['BsmtExposure'].fillna('None', inplace = True)

# We are going to fill BsmtFinType1 with none because no basement square footage
train['BsmtFinType1'].fillna('None', inplace = True)

# We are going to fill BsmtFinType2 with none because no basement square footage
train['BsmtFinType2'].fillna('None', inplace = True)

#train['Electrical'].value_counts()
#SBrkr    1334
#FuseA      94
#FuseF      27
#FuseP       3
#Mix         1

# We are going to fill Electrical with SBrkr because it is most common (see above)
train['Electrical'].fillna('SBrkr', inplace = True)

# We are going to fill FireplaceQu with None because there is 0 on Fireplaces
train['FireplaceQu'].fillna('None', inplace = True)

# We assume that the GarageType is None since there is 0 for all the garage values
train['GarageType'].fillna('None', inplace = True)

# We assume that the GarageYearBuilt is the same as the YearBuilt.
train['GarageYrBlt'].fillna(value = train['YearBuilt'], inplace = True)

# We assume that the GarageFinish is None because no garage
train['GarageFinish'].fillna('None', inplace = True)

# We assume that the GarageQual is None because no garage
train['GarageQual'].fillna('None', inplace = True)

# We assume that the GarageCond is None because no garage
train['GarageCond'].fillna('None', inplace = True)

# We are going to fill PoolQC with none because no pool
train['PoolQC'].fillna('None', inplace = True)

# We are going to fill Fence with none because no fence
train['Fence'].fillna('None', inplace = True)

# We are going to fill MiscFeature with none because description says none
train['MiscFeature'].fillna('None', inplace = True)

###############################################################################

# Check for NA's after filling.
# numna = train.isnull().sum()
# numna_col = numna[numna.values > 0]

# There are 0 NA's so we move on!

###############################################################################

# Get Dummies; One-hot encoding the categorical variables

train = pd.get_dummies(train)

# Set up Model

X = train.drop(columns = 'SalePrice')
y = train['SalePrice']

# Finally, let's divide the data into training and testing sets:

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the Model

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=50, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

# Evaluate the Model

from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# Feature Importance and Model Score
feature_importance = regressor.feature_importances_
score = regressor.score(X_train,y_train)

# Score = 0.9781396188591381














































