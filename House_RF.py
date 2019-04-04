# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 19:05:41 2019

Nate Jermain
Housing Prices 
"""

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

########## Exploratory Data Analysis ##############
test=pd.read_csv("C:/Users/w10007346/Dropbox/Kaggle/House_Comp/test.csv")
train=pd.read_csv("C:/Users/w10007346/Dropbox/Kaggle/House_Comp/train.csv")

# the response variable 
sns.distplot(train.SalePrice)
train.SalePrice.describe()
# how many features
len(train.columns.values)
# plot heatmap of features
import os
#set dir
os.chdir('C:/Users/w10007346/Dropbox/Kaggle/House_Comp')

mask = np.zeros_like(train.corr())
mask[np.triu_indices_from(mask)]=True
cmap = sns.diverging_palette(180, 30, as_cmap=True)
with sns.axes_style('white'):
    fig, ax = plt.subplots(figsize=(13,11))
    heat=sns.heatmap(train.corr(), vmax=.8, mask=mask, cmap=cmap, cbar_kws={'shrink':.5}, linewidth=.01);
       
fig=heat.get_figure()
#save
fig.savefig('heatfig.png', dpi=800, bbox_inches="tight")

########## Data Cleaning #############################

# identify test versus train
test.columns.values
test.drop("Id", axis = 1, inplace = True)
test['set']=np.tile('test',len(test))


train.drop("Id", axis = 1, inplace = True)
train['set']=np.tile('train',len(train))

# combine test and training datasets to remove NAs in one sweep
df=train.append(test, ignore_index=True)

df.columns.values

df.head()


#are there any NAs
df.isnull().values.any() # of course

#how many in each feature
nas=df.isnull().sum().sort_values(ascending=False)
hasnas=nas[nas!=0]
hasnas


# Utilities
df.Utilities.describe()
# all utilities are the same, so remove column
df=df.drop(['Utilities'], axis=1)

# MSZoning
df.MSZoning.describe()
pd.Series(df.MSZoning).value_counts().plot('bar')
# vastly most common is RL replace missing with RL
df.MSZoning=df['MSZoning'].fillna(df['MSZoning'].mode()[0])

# Lot frontage
df.LotFrontage.describe()
sns.distplot(df.LotFrontage.dropna())
df.Neighborhood.describe()
# replace nas with median values for lot frontage in the same neighborhood
df.LotFrontage=df.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

# Alleys
df.Alley.describe()
# NA means no alley access from data file
df.Alley=df.Alley.fillna("None")

# Exterior 1st and 2nd
df.Exterior1st=df.Exterior1st.fillna(df.Exterior1st.mode()[0])
df.Exterior2nd=df.Exterior2nd.fillna(df.Exterior2nd.mode()[0])

# MasVnrType and area
# no masonry veneer for areas with na and none for types
df.MasVnrType=df.MasVnrType.fillna('None')
df.MasVnrArea=df.MasVnrArea.fillna(0)

# Bsmts
for i in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    df[i] = df[i].fillna('None')
    
for i in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    df[i] = df[i].fillna(0)

# kitchen
df.KitchenQual=df.KitchenQual.fillna(df.KitchenQual.mode()[0])

# functional
df.Functional=df.Functional.fillna('Typ')

# fireplace
df.FireplaceQu=df.FireplaceQu.fillna('None')

# Garages categorical features to none
for i in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    df[i] = df[i].fillna('None')


# Garage continuous features to 0
for i in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    df[i] = df[i].fillna(0)


# Fences
df.Fence=df.Fence.fillna('None')

#Pool QC
df.PoolQC=df.PoolQC.fillna('None')

# Miscfeature
df.MiscFeature=df.MiscFeature.fillna('None')

# Sale Type
df.SaleType.describe()
df.SaleType=df.SaleType.fillna(df.SaleType.mode()[0])

# Electric system
df.Electrical.describe()
df.Electrical=df.Electrical.fillna(df.Electrical.mode()[0])

# check to make sure all NAs gone except sale price for test set
nas=df.isnull().sum()
hasnas=nas[nas!=0]
hasnas


df.columns.values

# Some features are numerical that represent categories
df.MSSubClass.describe()
df.MSSubClass=df.MSSubClass.apply(str)
df.OverallCond.describe()
df.OverallCond=df.OverallCond.astype(str)
df.YrSold.describe()
df.YrSold=df.YrSold.astype(str)
df.MoSold=df.MoSold.astype(str)

# store response and remove from aggregate data set
resp=train.SalePrice 
df=df.drop('SalePrice', axis=1)


########## check for skewed features #########################################
# plot sale price
sns.distplot(resp) # positively skewed 
resp=np.log1p(resp) # transform by log(1+x)
sns.distplot(resp) 

# transform features by a flexible box cox
num_feats=df.dtypes[df.dtypes!='object'].index
skew_feats=df[num_feats].skew().sort_values(ascending=False)
skewness=pd.DataFrame({'Skew':skew_feats})
skewness=skewness[abs(skewness)>0.75].dropna()
skewed_features=skewness.index

# add one to all skewed features 
df[skewed_features]+=1
# conduct boxcox transformation
from scipy.stats import boxcox

for i in skewed_features:
    df[i],lmbda=boxcox(df[i], lmbda=None)

# check to see if skew decreased    
num_feats=df.dtypes[df.dtypes!='object'].index
skew_feats=df[num_feats].skew().sort_values(ascending=False)
skewness=pd.DataFrame({'Skew':skew_feats})
skewness=skewness[abs(skewness)>0.75].dropna()
    
    
# dummy code categorical variables 
df=pd.get_dummies(df)
df.columns.values

#  subset train and test sets from df that now has transformed features
dumtest=df.loc[df['set_test']==1]
dumtrain=df.loc[df['set_train']==1]

# drop indices for set and train sets
dumtest=dumtest.drop(['set_test','set_train'], axis=1)
dumtrain=dumtrain.drop(['set_test','set_train'], axis=1)

##################  Modeling  ###################################

############## Random Forest Regressor ##########################
from sklearn.ensemble import RandomForestRegressor

# the model prior to hyperparameter optimization
RFR=RandomForestRegressor(random_state=1)

#### use random search to identify the best hyperparameters using Kfold CV
# number of trees 
n_estimators=[int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# max number of features to consider at every split
max_features = ['auto', 'sqrt', 'log2']
# max number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# grid to feed gridsearch
grid_param = {'n_estimators': n_estimators,
              'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

# Random search training
from sklearn.model_selection import RandomizedSearchCV
RFR_random = RandomizedSearchCV(estimator = RFR, param_distributions = grid_param, n_iter = 500,
                               cv = 5, verbose=2, random_state=42, n_jobs = -1)

RFR_random.fit(dumtrain, resp) 
print(RFR_random.best_params_)

# {'n_estimators': 2000, 'min_samples_split': 2, 'min_samples_leaf': 1, 
# 'max_features': 'sqrt', 'max_depth': None, 'bootstrap': False}


Best_RFR=RandomForestRegressor(n_estimators=2000, min_samples_split=2, min_samples_leaf=1,
                               max_features='sqrt', max_depth=None, bootstrap=False)


# use root mean squared error to measure accuracy of model on through cross validation
from sklearn.model_selection import KFold, cross_val_score
n_folds=5
def rmse_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(dumtrain)
    rmse= np.sqrt(-cross_val_score(model, dumtrain, resp, scoring="neg_mean_squared_error", cv = kf))
    return(rmse.mean())
    
rmse_cv(Best_RFR) # 0.1499


############# XG Boosting Regression ################################
from xgboost import XGBRegressor

Boost = XGBRegressor(random_seed=1)

# optimize hyperparameters
#### use random search to identify the best hyperparameters using Kfold CV
# learning rate

# number of trees 
n_estimators=[int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# percentage of samples per tree
subsample = [.6,.7,.8,.9,1]
# max number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 50, num = 10)]
#  minimum sum of weights of all observations required in a child
Min_child_weight = [1,3,5,7]
# percentage of features used per tree
colsample_bytree=[.6,.7,.8,.9,1]
# learning rate or step size shrinkage
learning_rate=[.01,.015,.025,.05,.1]
# Minimum loss reduction required to make a split
gamma = [.05,.08,.1,.3,.5,.7,.9,1]
# grid to feed gridsearch
rand_param = {'n_estimators': n_estimators,
              'subsample': subsample,
               'max_depth': max_depth,
               'colsample_bytree': colsample_bytree,
               'min_child_weight': Min_child_weight,
               'learning_rate': learning_rate,
               'gamma': gamma}

Boost_random = RandomizedSearchCV(estimator = Boost, param_distributions = rand_param, n_iter = 500,
                               cv = 5, verbose=2, random_state=42, n_jobs = -1)

Boost_random.fit(dumtrain, resp) 
print(Boost_random.best_params_)
# {'subsample': 0.6, 'n_estimators': 1800, 'min_child_weight': 3, 'max_depth': 50, 
# 'learning_rate': 0.015, 'gamma': 0.05, 'colsample_bytree': 0.6}

# model with the optimum hyperparameters
Best_Boost = XGBRegressor(subsample=.6, n_estimators=1800, min_child_weight=3, max_depth=50,
                          learning_rate=.015, gamma=.05, colsample_bytree=.6)
# evaluate rmse
rmse_cv(Best_Boost) #0.1308

# use exponential function to get appropriately scaled sale prices
Best_Boost.fit(dumtrain,resp)
ypred=np.expm1(Best_Boost.predict(dumtest))

sub=pd.DataFrame()
test=pd.read_csv("C:/Users/w10007346/Dropbox/Kaggle/House_Comp/test.csv")
sub['Id']=test['Id']
sub['SalePrice']=ypred
sub.to_csv('KaggleSub.csv', index=False)
