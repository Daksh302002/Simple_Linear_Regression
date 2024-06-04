import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Importing the dataset
dataset = pd.read_csv("Salary_Data.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Printing X and y
# print(X)
# print(y)

# Splitting the dataset into traning and test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Printing X_train, X_test, y_train, y_test
# print(X_train)
# print(X_test)
# print(y_train)
# print(y_test)


# Training the Simple Linear Regression model on the Traning set
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)  # y_pred contains predicted Salary

# Visulising the training set results
plt.scatter(X_train, y_train, color="red")
plt.plot(X_train, regressor.predict(X_train), color="blue")
plt.title("YearsExperience vs Salary (Training set)")
plt.xlabel("YearsExperience")
plt.ylabel("Salary")
plt.show()

# Visulising the test set results
plt.scatter(X_test, y_test, color="red")
plt.plot(X_train, regressor.predict(X_train), color="blue")
plt.title("YearsExperience vs Salary (Test set)")
plt.xlabel("YearsExperience")
plt.ylabel("Salary")
plt.show()
