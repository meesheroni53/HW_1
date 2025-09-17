import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

#1) Create a dataset with 10,000 rows and 4 random variables: 2 of them normally distributed, 2 uniformly distributed
# Set random seed for reproducibility
np.random.seed(10)

# Create a dataset with 10,000 rows
num_rows = 10000

# Create 2 normal distribution variables
x1 = np.random.normal(loc=0, scale=1, size=num_rows)
x2 = np.random.normal(loc=0, scale=1, size=num_rows)

# Create 2 uniform distribution variables
x3 = np.random.uniform(low=0.0, high=1.0, size=num_rows)
x4 = np.random.uniform(low=0.0, high=1.0, size=num_rows)

#2) add another variable ("y") as a linear combination, with some 
# coefficients of your choice, of: the 4 variables above; the square 
# of one of the variables; and some random "noise" (randomly distributed, 
# centered at 0, small variance). Hint: think about y = a1*x1+a2*x2+a3*x3+a4*x4+epsilon. 
# You will get a dataset with 5 variables.

# Define the coefficients for the linear combination
a1 = 3.5
a2 = -1.5
a3 = 5.0
a4 = 2.2
a5 = 0.4 # Coefficient for the square term

# Adding random noise (small variance of 0.15)
epsilon = np.random.normal(loc=0, scale=0.15, size=num_rows)

# Calculate y as y = a1*x1+a2*x2+a3*x3+a4*x4+epsilon
y = (a1*x1)+(a2*x2)+(a3*x3)+(a4*x4)+(a5*x4**2)+epsilon

# Create the dataframe
df = pd.DataFrame({
    'normal_variable1': x1,
    'normal_variable2': x2,
    'uniform_variable1': x3,
    'uniform_variable2': x4,
    'uniform_variable2_squared': x4**2,
    'y': y
})

# Preview the dataset
print(df.head())

#3) Split the dataset in #2 into 70% for training and 30% for testing
# Splitting data
train, test = train_test_split(df, test_size=0.3, random_state=12)

print("Train Set:\n", train)
print("\nTest Set:\n", test)

#4) estimate the linear regression coefficients using OLS for the training data; 
# compute the Mean Standard Error (average of squared differences between estimates 
# and actuals) on both the training dataset, and the testing dataset

# Defining X Variables and target (y)
cols = ['normal_variable1', 'normal_variable2', 'uniform_variable1', 'uniform_variable2','uniform_variable2_squared']
X_train = train[cols]
y_train = train['y']
X_test = test[cols]
y_test = test['y']

print(X_train.head())

# Fit model with the training data
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on both the training and testing data to calculate the MSE
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Compute the mean squared error for both the training and testing data
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)

# Output results
coefs = np.round(np.append(model.intercept_, model.coef_), 2)
formatted_coefs = [f"{coef:.2f}" for coef in coefs]
print("Estimated coefficients:", formatted_coefs)
print("Training MSE:", np.round(mse_train,4))
print("Testing MSE:", np.round(mse_test,4))

#5) Use bootstrapping to create 10 other samples from the data your created in #2 above
# Store coefficients from each bootstrap

# Creates an empty list for the bootstrap loop
bootstrap_coefs = []

# Creates the bootstrap loop (10 samples based off original dataframe)
for i in range(10):
    bootstrap_sample = df.sample(n=len(df), replace=True) 
    
    X_boot = bootstrap_sample[cols]
    y_boot = bootstrap_sample['y']

#6) Estimate the linear regression coefficients using OLS for 
# each of the 10 bootstrap samples in #5    
    model = LinearRegression()
    model.fit(X_boot, y_boot)
    
    coefs = np.append(model.intercept_, model.coef_)
    bootstrap_coefs.append(coefs)

coef_df = pd.DataFrame(bootstrap_coefs, columns=['Intercept'] + cols)

# Display the results
print("Bootstrapped Coefficients (10 samples):")
print(np.round(coef_df),6)

#7) For each linear regression parameter, use the estimates 
# computed in #6 to get the mean and the standard deviation.

# Calculate the mean & standard deviation of coefficients
print("\nMean of Bootstrapped Coefficients:")
print(coef_df.mean())

print("\nStandard Deviation of Bootstrapped Coefficients:")
print(coef_df.std())

#8) What can you say about the coefficients in #4 looking at the results in #7?
# Define the variables from question #4
original_coefs = [0, a1, a2, a3, a4, a5] 
index = ['Intercept', 'normal_variable1', 'normal_variable2', 'uniform_variable1', 'uniform_variable2', 'uniform_variable2_squared']

# Create the Series to compare the original with the revised results
original = pd.Series(data=original_coefs, index=index)
mean_results=pd.Series(data=coef_df.mean(), index=index)

difference=original-mean_results
print("\nDifference between original coefficients in #4 and results in #7:")
print(difference)

#Here we can see the difference between the mean of the bootstrapping samples and the original values
# is near 0, showing that the bootstrapping produced estimates close to the original estimates. It  shows
# that our estimates are reliable and there is low bias in the estimates.

#https://medium.com/@okon.judith/create-datasets-yourself-fa6596d03778,  
#https://www.geeksforgeeks.org/numpy/normal-distribution-in-numpy/,
#https://numpy.org/doc/2.2/reference/random/generated/numpy.random.uniform.html, 
#https://medium.com/@whyamit404/understanding-train-test-split-in-pandas-eb1116576c66,
#https://www.geeksforgeeks.org/machine-learning/python-linear-regression-using-sklearn/, 
#https://www.datacamp.com/tutorial/sklearn-linear-regression
