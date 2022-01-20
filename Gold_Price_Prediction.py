# Gold Price Prediction
# Dataset: GoldPrice
# Random Forest Regressor



# Import Libraries
import numpy as np  # Used for Array
import pandas as pd # Used for DataFrame
import matplotlib.pyplot as plt  # Used for Data Visualization
import seaborn as sns  # Used for Data Visualization
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics  # To find error score, accuracy score


# Load the dataset
path = "E:/Kaggle/Gold Price Prediction/Dataset/gold_price.csv"
gold_data = pd.read_csv(path)


# First 5 rows of data
gold_data.head()

# SPX --> Stock Price
# GLD --> Gold Price
# USO --> United States Oil Price
# SLV --> Silver Price
# EUR/USD --> European Price / United States Dollar


# Last 5 rows of data
gold_data.tail()


# Number of rows and columns :-
gold_data.shape


# Getting some basic information about the data
gold_data.info()


# Checking the number of missing values
gold_data.isnull().sum()


# Getting some statistical measures of the data
gold_data.describe()


# Correlation
# 1. Postive Correlation
# 2. Negative Correlation
# Whenever we are working on Regression Data we should always check correlation
correlation = gold_data.corr()


# Constructing a heatmap to understand the correlation
plt.figure(figsize = (8,8))
sns.heatmap(correlation, cbar = True, fmt = '.1f', annot = True, annot_kws = {'size':8}, cmap = 'Blues')
# fmt = '.1f' means we want one floating point
# Name of the columns mean 'annotation / annot'


# Correlation values of GLD 
print(correlation['GLD'])


# Checking the distribution of Gold Price
sns.distplot(gold_data['GLD'], color = 'Green')


# Splitting the Features and Target
X = gold_data.drop(['Date', 'GLD'], axis = 1)
Y = gold_data['GLD']

print(X)
print(Y)


# Splitting into Training and Testing Data
x_train,x_test,y_train,y_test = train_test_split(X, Y, test_size = 0.2, random_state = 2)


# Model Training
# Random Forest
regressor = RandomForestRegressor(n_estimators = 100)
regressor.fit(x_train, y_train)


# Model Evaluation
# Prediction on Test Data
test_data_prediction = regressor.predict(x_test)
print(test_data_prediction)


# R Squared Error
error_score = metrics.r2_score(y_test, test_data_prediction)
print("R Squared Error :- ", error_score)


# Compare the Actual Values and Predicted Values in a Plot
y_test = list(y_test)
plt.plot(y_test, color = 'red', label = 'Actual Value')
plt.plot(test_data_prediction, color = 'green', label = 'Prediction Value')
plt.title('Actual Value VS Predicted Value')
plt.xlabel('No of Values')
plt.ylabel('GLD Price')
plt.legend()
plt.show()