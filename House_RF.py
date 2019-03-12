# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 19:05:41 2019

Nate Jermain
Housing Prices 
"""

import pandas as pd
import numpy as np
import seaborn as sns

########## Data Cleaning #############################
test=pd.read_csv("C:/Users/w10007346/Dropbox/Kaggle/House_Comp/test.csv")
#get rid of id column
test.drop("Id", axis = 1, inplace = True)
# identify test versus train
test['set']=np.tile('test',len(test))

train=pd.read_csv("C:/Users/w10007346/Dropbox/Kaggle/House_Comp/train.csv")
train.drop("Id", axis = 1, inplace = True)
train['set']=np.tile('train',len(train))

# combine test and training datasets to remove NAs in one sweep

df=train.append(test, ignore_index=True)


df.columns.values

df.head()

sns.distplot(df.SalePrice.dropna())

#are there any NAs
df.isnull().values.any() # of course

#how many in each feature
nas=df.isnull().sum()
hasnas=nas[nas!=0]
hasnas


# Utilities
df.Utilities.describe()
# all utilities are the same, so remove column
df=df.drop(['Utilities'], axis=1)

# MSZoning
df.MSZoning.describe()
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



# get rid of needless columns
df=df.drop('set', axis=1)
resp=train.SalePrice # store response
df=df.drop('SalePrice', axis=1)


# dummy code categorical variables 
df=pd.get_dummies(df)
dumtrain=df[:train.shape[0]]
dumtest=df[:test.shape[0]]

# plot sale price
sns.distplot(resp) # positively skewed 
resp=np.log1p(resp) # transform by log(1+x)
sns.distplot(resp) 

# split training data into validation and train
from sklearn.model_selection import train_test_split
#train_X, val_X, train_y, val_y = train_test_split(dumtrain.drop('SalePrice',axis=1), dumtrain.SalePrice, random_state = 0)

############## Random Forest Regressor ##########################
from sklearn.ensemble import RandomForestRegressor

#model=RandomForestRegressor(random_state=1)
#model.fit(train_X, train_y)
#
#val_predictions=model.predict(val_X)
#
#from sklearn.metrics import mean_absolute_error
#print(mean_absolute_error(val_y, val_predictions)) # 0.10056
#
#results=pd.concat([val_y,pd.Series(val_predictions)], axis=1).dropna()
#results.columns.values
#results.columns=['Observed', 'Predicted']
#results.plot.scatter('Observed', 'Predicted')
# results are fairly poor


########## check for skewed features #########################################
num_feats=df.dtypes[df.dtypes!='object'].index

skew_feats=df[num_feats].skew().sort_values(ascending=False)
skewness=pd.DataFrame({'Skew':skew_feats})
skewness=skewness[abs(skewness)>0.75].dropna()

from scipy.special import boxcox1p
skewed_features=skewness.index
lam=0.15

for i in skewed_features:
    df[i]=boxcox1p(df[i],lam)

######## Try random forest regressor with transformed data ############
#  subset train and test sets from df that now has transformed features
dumtrain=df[:train.shape[0]]
dumtest=df[:test.shape[0]]

len(dumtest.columns.values)
len(dumtrain.columns.values)


# plot sale price
sns.distplot(resp) # no longer skewed


############ Now use Kfold cross validation  ################
from sklearn.model_selection import KFold, cross_val_score, train_test_split

n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(dumtrain)
    rmse= np.sqrt(-cross_val_score(model, dumtrain, resp, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)

# random forest 
RFR=RandomForestRegressor(random_state=1)
score=rmsle_cv(RFR)
score.mean() #.15

# gradient boosting regression
from sklearn.ensemble import GradientBoostingRegressor
Boost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)

score=rmsle_cv(Boost)
score.mean() #.12

# fit the training dataset on the model
BoostMd=Boost.fit(dumtrain,resp)

# predictions for test set
np.expm1(BoostMd.predict(dumtest))


