import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
X = load_iris().data
y = load_iris().target
from sklearn.model_selection import train_test_split

X[0],y[0]

dir(load_iris())

load_iris().feature_names

load_iris().target_names

setosa = X[y==0]
versicolor = X[y==1] 
virginica = X[y==2]

plt.scatter(setosa[:, 0], setosa[: ,2])
plt.scatter(versicolor[:,0], versicolor[:,2])
plt.scatter(virginica[:,0], virginica[:,2])

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 11 )

import numpy as np
from random import randrange
from math import sqrt
class KNearestNeighbor(object):
    
    def __init__(self, k):
        
        self.k = k
        #v1=x_train,v2=test""""""
    
    def euclidean_distance(self, v1, v2):
        
        distance = 0
        
        for i in  range(len(v2)):
            distance += (v1[i] - v2[i])**2
        return sqrt(distance)
    
    
        
    
    def predict(self, x_train, test_instance):
        
        distances = []
        
        for train_row in range(len(x_train)):
            
            distance = self.euclidean_distance(x_train[train_row ], test_instance)
            distances.append((y_train[train_row], distance))
        
        distances.sort(key = lambda t:t[1])
        #print(distances[0][0])
        
        neighbors = []
        
        for i in range(self.k): 
            
            neighbors.append(distances[i][0])
        #print(neighbors)     
      
        
        prediction = max( set(neighbors) , key = neighbors.count) 
        return prediction

    
    def Evaluate(self, pred, y_test):
    
        predicted = pred
        actual = y_test

        count = 0
        for i,j in zip( predicted, actual):
            if i != j:
                count += 1

        print("Score : ", (count/len(actual)))  
        print("Error : ", 1-(count/len(actual)))

knn  = KNearestNeighbor( k = 13)

pred = []
for i in x_test :
    prediction = knn.predict(x_test, i)
    pred.append(prediction)

knn.Evaluate(pred, y_test)









