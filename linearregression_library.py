import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from random import randrange

from sklearn.model_selection import train_test_split
data = pd.read_csv("headbrain.csv")
X, Y = data["Head Size(cm^3)"], data["Brain Weight(grams)"]
X, Y = np.array(data["Head Size(cm^3)"]),np.array( data["Brain Weight(grams)"])
X = X.reshape(-1, 1)

class LinearRegression():
    
    def __init__(self):
        pass
    
        
    def fit(self, X, Y):

        mean_x = np.mean(X)
        mean_y = np.mean(Y)

        # Total number of values
        n = len(X)

        # Using the formula to calculate m and c
        numer = 0
        denom = 0
        for i in range(n):
            numer += (X[i] - mean_x) * (Y[i] - mean_y)
            denom += (X[i] - mean_x) ** 2
            m = numer / denom 
            c = mean_y - (m * mean_x)
        # Print coefficients
        return c, m
    
    
    

    def train_test_split(self, dataset1, dataset2, split):
        train_l = []
        test_l = []
        for i in (dataset1, dataset2):
            train = []
            train_size = split * len(i)
            test = list(i)
            while len(train)< train_size:
                index = randrange(len(test))
                train.append(test.pop(index))

            train_l.append(list(train))
            test_l.append(list(test))

        return train_l[0], test_l[0], train_l[1], test_l[1]
    
    def predict(self, c, m, X):
        
        # Plotting Values and Regression Line
        max_x = np.max(X) + 100
        min_x = np.min(X) - 100
        
        # Calculating line values x and y
        x = np.linspace(min_x, max_x, 1000)
        y = c + m * x
        
        return x, y
        
    def plot(self, x, y, X, Y):
        # Ploting Line
        plt.plot(x, y, color='#52b920', label='Regression Line')
        # Ploting Scatter Points
        plt.scatter(X, Y, c='#ef4423', label='Scatter Plot')

        plt.xlabel('Head Size in cm3')
        plt.ylabel('Brain Weight in grams')
        plt.legend()
        plt.show()

model = LinearRegression()
X_train, x_test, y_train, y_test = model.train_test_split(X,Y, 0.7)
c, m = model.fit(X_train, y_train) 
x, y = model.predict(c, m, x_test)
model.plot(x, y, X_train, y_train)

from math import sqrt

actual = y_test
predicted = y
sse = 0
for i,j in zip(actual, predicted):
    
    error = i-j
    sqe = error**2
    sse = sse + sqe
mse = sse/len(actual)
rmse = sqrt(mse)

rmse

ss_tot = 0
ss_res = 0
for i in range(len(x_test)):
    y_pred = c + m * X[i]
    ss_tot += (Y[i] - np.mean(y)) ** 2
    ss_res += (Y[i] - y_pred) ** 2
r2 = 1 - (ss_res/ss_tot)
print("R2 Score")
print(r2)





