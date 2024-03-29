
############################################################################################
#Course: CS680
#Name: Chris Binoi Verghese
#Course: MDSAI
#Description: For d = 1 dataset, computes the(unregularized) least squares linear regression and the k-NN regression solution
############################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Function to plot the linear and kNN regression data onto a single plot for each dataset
def plot_reg(X_test, y_test, X, y_pred_linear,knn, dataset_name):
    plt.figure(figsize=(16, 6))
    plt.scatter(X_test, y_test, color='red', label='Actual')
    plt.plot(X.reshape(-1, 1), y_pred_linear.reshape(-1, 1), color='blue', label='Least Squares')
    plt.scatter(X.reshape(-1, 1), knn[0].reshape(-1, 1), label=f'KNN k={1}')
    plt.scatter(X.reshape(-1, 1), knn[1].reshape(-1, 1), label=f'KNN k={9}')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.title(f'Regression Predictions for {dataset_name}')

#Function to plot the MSE of the linear and kNN regression data onto a single plot for each dataset
def plot_mse(k, mse_knn, mse_lin, name):
    plt.figure(figsize=(12, 6))
    plt.plot(k, mse_knn, marker='o', color='blue', label='K-NN')
    plt.axhline(y=mse_lin, color='red', label='Least Squares Linear Regression')
    plt.xlabel('k')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.title(f'Mean Squared Error of Test Data for {name}')

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
        
    #Accepts X data and calculates a suitable predicted y based on training data
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

#Function to get coefficients for a linear regression model and return predicted y for test data
def least_squares_linear_regression(X, y, test):
    # Add a column of ones to X for the bias term
    Xt = np.hstack((X, np.ones([X.shape[0],1])))
    # Compute the coefficients using the least squares function
    coeff = np.linalg.lstsq(Xt, y, rcond=None)[0]
    #Calculating and return the predicted y
    y_pred =  test*coeff[:-1] + coeff[-1]
    return y_pred

#Initialize all variables
y_D_kNN = [1,1,1,1,1,1,1,1,1]
y_E_kNN = [1,1,1,1,1,1,1,1,1]
y_D_kNN_graph = [1,1]
y_E_kNN_graph = [1,1]
mse_knn_D = [1,1,1,1,1,1,1,1,1]
mse_knn_E = [1,1,1,1,1,1,1,1,1]
# Load training and test data for D and E from excel files
X_train_D = pd.read_csv("a1-files/X_train_D.csv").values
Y_train_D = pd.read_csv('a1-files/Y_train_D.csv').values
X_train_E = pd.read_csv("a1-files/X_train_E.csv").values
Y_train_E = pd.read_csv('a1-files/Y_train_E.csv').values
X_test_D = pd.read_csv("a1-files/X_test_D.csv").values
Y_test_D = pd.read_csv('a1-files/Y_test_D.csv').values
X_test_E = pd.read_csv("a1-files/X_test_E.csv").values
Y_test_E = pd.read_csv('a1-files/Y_test_E.csv').values

# Perform Linear Regression for D and E and get their predicted y for test data
y_D_lin = least_squares_linear_regression(X_train_D, Y_train_D, X_test_D)
y_E_lin = least_squares_linear_regression(X_train_E, Y_train_E, X_test_E)
#Calculate the MSE of the predicted y values obtained through linear regression
D_mse_lin = ((Y_test_D - y_D_lin) ** 2).mean()
E_mse_lin = ((Y_test_E - y_E_lin) ** 2).mean()

#Calculate the predicted y values for test data for k = 1 to 9
for i in range(9):
    knn = KNNRegression(i+1,X_train_D, Y_train_D)
    y_D_kNN[i] = knn.predict(X_test_D)
    knn = KNNRegression(i+1,X_train_E, Y_train_E)
    y_E_kNN[i] = knn.predict(X_test_E)
    # Calculate MSE for all values of k for predicted y values
    mse_knn_D[i] = ((Y_test_D - y_D_kNN[i]) ** 2).mean()
    mse_knn_E[i] = ((Y_test_E - y_E_kNN[i]) ** 2).mean()

#For plotting reason plot all predicted data for x values between -1 and 1
X = np.linspace(-1, 1, 1000)  # 1000 points for the plot
# Make predictions for the x values using linear regression
y_D_lin = least_squares_linear_regression(X_train_D, Y_train_D, X)
y_E_lin = least_squares_linear_regression(X_train_E, Y_train_E, X)
# Make predictions for the x values using k Nearest Neighbour
knn_graph = KNNRegression(1,X_train_D, Y_train_D)
y_D_kNN_graph[0] = knn_graph.predict(X)
knn_graph = KNNRegression(9,X_train_D, Y_train_D)
y_D_kNN_graph[1] = knn_graph.predict(X)
knn_graph = KNNRegression(1,X_train_E, Y_train_E)
y_E_kNN_graph[0] = knn_graph.predict(X)
knn_graph = KNNRegression(9,X_train_E, Y_train_E)
y_E_kNN_graph[1] = knn_graph.predict(X)

# Plot results for Dataset D
plot_reg(X_test_D, Y_test_D, X ,y_D_lin, y_D_kNN_graph, "Dataset D")
# Plot results for Dataset E
plot_reg(X_test_E, Y_test_E, X,y_E_lin,y_E_kNN_graph, "Dataset E")
# Plot MSE for Dataset D
plot_mse(range(1, 10), mse_knn_D, D_mse_lin , "Dataset D")
# Plot MSE for Dataset E
plot_mse(range(1, 10), mse_knn_E, E_mse_lin, "Dataset E")
plt.show()
