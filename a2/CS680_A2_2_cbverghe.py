
############################################################################################
#Course: CS680
#Name: Chris Binoi Verghese
#Course: MDSAI
#Assignmnent 2 Question 2
#Description: Implements SVM regression using gradients
############################################################################################

import numpy as np
import statsmodels.api as sm 
import matplotlib.pyplot as plt
import sklearn.svm as svm

def data_prep(file_name1 , file_name2):
    #Read data from the given files and format them
    X_train = np.genfromtxt(file_name1, delimiter=',')
    Y_train = np.genfromtxt(file_name2, delimiter=',').reshape(-1)      
    #Standardize all read data
    X_train = (X_train - np.mean(X_train, axis=0))/np.std(X_train, axis=0)
    return X_train, Y_train

#Function to implement gradient descent algorithm
def SV_regression(X, y, C,e , learning_rate = 0.0001, max_pass  = 10000):
    w = np.zeros(np.shape(X)[1]).reshape(-1,1)  # Initialize coefficients to zeros
    b = 0
    predictions = []
    for iteration in range(1, max_pass):
        for k in range(len(y)):
            yi = y[k] - np.dot(X[k], w) + b
            if abs(yi) >= e:
                #Calculate the gradient of weight
                wgradient = np.where(yi > 0, -C*X[k], C*X[k])
                #Calculate the gradient of bias
                bgradient = np.where(yi > 0, -C, C)
                #Update w and b using gradient
                w -= learning_rate * wgradient.reshape(-1,1)
                b -= learning_rate * bgradient
            #The Proximal Step
            w = w/(1+learning_rate) 
            prediction = np.dot(X[k],w) + b #Calculating predictions
            predictions.append(prediction)

    # Calculate training error (MAE)
    training_error = np.sum([max(0, abs(y[i] - predictions[i]) - e) for i in range(len(y))])
    # Calculate training loss 
    training_loss = np.sum([max(0, abs(y[i] - predictions[i]) - e) for i in range(len(y))]) + 0.5*np.linalg.norm(w)**2
    # Report the results
    print("Training Error (MAE):", training_error)
    print("Training Loss:", training_loss)
    return w,b



#Read all the data from dataset C
X_train_C,Y_train_C = data_prep('a2-files\data\X_train_C.csv', 'a2-files\data\Y_train_C.csv')
X_test_C,Y_test_C = data_prep('a2-files\data\X_test_C.csv', 'a2-files\data\Y_test_C.csv')
#Get the weight and bias from SVM regression
w,b = SV_regression(X_train_C, Y_train_C, 1,0.5)
#Calculate prediction of testing data
prediction = np.dot(X_test_C,w) + b
#Calculate and print the testing error
testing_error = np.sum([max(0, abs(Y_test_C[i] - prediction[i]) -0.5) for i in range(len(Y_test_C))])
print("Testing error:", testing_error)

