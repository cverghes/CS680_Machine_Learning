
############################################################################################
#Course: CS680
#Name: Chris Binoi Verghese
#Course: MDSAI
#Description: Implement k-nearest neighbour regression with time complexity O(nd)
############################################################################################

import numpy as np

#Function to find the l2 distance between two points
def l2(X_train, X_test):
    x1 = np.array(X_train)
    x2 = np.array(X_test)
    dist = np.sqrt(np.sum((x1 - x2) ** 2))
    return dist

#FQuickselect algorithm to find n loest number to achieve O(nd)
def quickselect(arr, n):
    #Return if input array has only 1 variable
    if len(arr) == 1:
        return arr[0]
    #Initialize lists 
    lows = []
    highs = []
    pivots = []
    #Set pivot as middle value of the array
    pivot = arr[len(arr)//2]
    #Search through array to seperate elements based pivot
    for x in arr:
       if x < pivot:
           lows.append(x)
       elif x > pivot:
           highs.append(x)
       else:
           pivots.append(x)
    #If objective fullfilled return else recurse into the function ith a different arr and n
    if n < len(lows):
        return quickselect(lows, n)
    elif n < len(lows) + len(pivots):
        return pivots[0]
    else:
        return quickselect(highs, n - len(lows) - len(pivots))
    
#Class to store data and functions related to kNN
class KNNRegression:
    #Initialializes all model training data
    def __init__(self, k,X,y):
        self.k = k
        self.X_train = X
        self.Y_train = y

    #Accepts X data and calculates a suuitable predicted y based on training data
    def predict(self, X_test):
        y_pred = []
        for x in X_test:
            #Initialize loop variables
            count = 0
            k_nearest_labels = 0
            #Creates a list of all distances between the test point and training data
            dist = [l2(x_train, x) for x_train in self.X_train]
            #Use quickselect to find the kth smallest value
            kth_smallest_distance = quickselect(dist, self.k)
            #runs through the list of distances to find the k least ones and add them
            for i in range(len(dist)):
                if(dist[i]<=kth_smallest_distance):
                    count = count + 1
                    k_nearest_labels += self.Y_train[i]
                if(count==self.k):
                    break
            #Add the average of y values of the k nearest neighbours to list of predicted y
            y_pred.append(k_nearest_labels/count)
        return np.array(y_pred)

'''

# Example test case:
k = 2
X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
Y_train = np.array([10, 20, 30, 40])
knn = KNNRegression(k,X_train, Y_train)
X_test = np.array([[2, 2], [3, 3]])
y_pred = knn.predict(X_test)
print(y_pred) 
print(quickselect(Y_train, k))
'''