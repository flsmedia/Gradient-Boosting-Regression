# Gradient-Boosting-Regression
I used gradient boosting regression and random forest regression to predict real estate prices for the kaggle competition titled "House Prices: Advanced Regression Techniques" 

![fig1](https://github.com/njermain/Gradient-Boosting-Regression/blob/master/architecture-beautiful-exterior-106399%20(1).jpg)

## Data Cleaning
Data: The data for this analysis can be found at https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data

There are 37 features in the dataset that describe physical attributes of each house.

![fig2](https://github.com/njermain/Gradient-Boosting-Regression/blob/master/HomePrices.JPG)

Figure 1: Heat map showing the correlation among features

This dataset required a lot of cleaning due to the number of NAs:

![fig2](https://github.com/njermain/Gradient-Boosting-Regression/blob/master/HousePrices2.JPG)

I filled in NAs for features where it was logical to do so, and removed those that had more than 80% missing. 

Additionally many features were skewed, including the response variable Price:

![fig2](https://github.com/njermain/Gradient-Boosting-Regression/blob/master/HousePrices3.JPG)

I used a box cox transformation on skewed features, and a log transformation on the response variable. 

## Modeling

I applied both a gradient boosting regression model and a random forest regressor to predict price. 

Hyperparameters were optimized using RandomGridSearch. For the gradient boosting regression model, I optimized:

![fig2](https://github.com/njermain/Gradient-Boosting-Regression/blob/master/HomePrices4.JPG)

I optimized the following hyperparameters for the random forest regressor:

![fig2](https://github.com/njermain/Gradient-Boosting-Regression/blob/master/HomePrices5.JPG)

The two models were compared given cross validation scores; the gradient boosting regressor had superior performance. 

The gradient boosting regression model performed with a RMSE value of 0.1308 on the test set, not bad!

Additional explanations and code can be found at:
https://towardsdatascience.com/home-value-prediction-2de1c293853c?source=friends_link&sk=704e52c7130e767aa32d6f6bae9678b9



