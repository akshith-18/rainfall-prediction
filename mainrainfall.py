#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the data
data = pd.read_csv("/Users/akshithpallerla/Downloads/district wise rainfall normal.csv")

# Print data types to check for non-numeric columns
print("Data types:")
print(data.dtypes)

# Convert columns to numeric if necessary
data = data.apply(pd.to_numeric, errors='coerce')

# Fill missing values with the mean of each column
data = data.fillna(data.mean())

# Print the number of missing values in each column to confirm
print("\nMissing values per column after filling:")
print(data.isnull().sum())

# Flatten the monthly and yearly data into features and target
df = data.melt(id_vars=['STATE_UT_NAME', 'DISTRICT'], var_name='Month', value_name='Avg_Rainfall')

# Map month names to numerical values
Month_map = {'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6, 'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12}
df['Month'] = df['Month'].map(Month_map)

# Ensure there are no missing values in the features or target
print("\nMissing values in features and target:")
print(df[['Month', 'Avg_Rainfall']].isnull().sum())

# Remove rows with missing values in features or target
df = df.dropna(subset=['Month', 'Avg_Rainfall'])

# Define features and target
X = np.array(df[['Month']])
y = np.array(df['Avg_Rainfall'])

# Check if there are any missing values in X or y
print("\nMissing values in features and target after dropping:")
print(pd.DataFrame(X).isnull().sum())
print(pd.Series(y).isnull().sum())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

# Initialize and train the models
linear_model = LinearRegression().fit(X_train, y_train)
lasso_model = Lasso(alpha=100.0).fit(X_train, y_train)
ridge_model = Ridge(alpha=100.0).fit(X_train, y_train)
svr_model = SVR(kernel='rbf').fit(X_train, y_train)
rf_model = RandomForestRegressor(max_depth=100, n_estimators=800).fit(X_train, y_train)

# Define the input data (month)
input_data = np.array([[9]])  # For September

# Predict using each model
pred_linear = linear_model.predict(input_data)
pred_lasso = lasso_model.predict(input_data)
pred_ridge = ridge_model.predict(input_data)
pred_svr = svr_model.predict(input_data)
pred_rf = rf_model.predict(input_data)

# Print the predictions
print(f'\nLinear Regression Prediction: {pred_linear[0]:.2f} mm')
print(f'Lasso Regression Prediction: {pred_lasso[0]:.2f} mm')
print(f'Ridge Regression Prediction: {pred_ridge[0]:.2f} mm')
print(f'SVR Prediction: {pred_svr[0]:.2f} mm')
print(f'Random Forest Prediction: {pred_rf[0]:.2f} mm')

# Evaluate the models
def evaluate_model(model, X_train, y_train, X_test, y_test):
    y_train_predict = model.predict(X_train)
    y_test_predict = model.predict(X_test)
    
    print(f'\n{model.__class__.__name__} Evaluation:')
    print(f'Train MSE: {mean_squared_error(y_train, y_train_predict):.2f}')
    print(f'Test MSE: {mean_squared_error(y_test, y_test_predict):.2f}')
    print(f'Train R2: {r2_score(y_train, y_train_predict):.2f}')
    print(f'Test R2: {r2_score(y_test, y_test_predict):.2f}')

# Evaluate each model
evaluate_model(linear_model, X_train, y_train, X_test, y_test)
evaluate_model(lasso_model, X_train, y_train, X_test, y_test)
evaluate_model(ridge_model, X_train, y_train, X_test, y_test)
evaluate_model(svr_model, X_train, y_train, X_test, y_test)
evaluate_model(rf_model, X_train, y_train, X_test, y_test)

