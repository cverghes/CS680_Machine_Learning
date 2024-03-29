
############################################################################################
#Course: CS680
#Name: Chris Binoi Verghese
#Course: MDSAI
#Description: Implements the gradient descent algorithm for solving ridge regression.
############################################################################################


import numpy as np
import time
import matplotlib.pyplot as plt

#Function to calculate the error 
def CalcError(X,y,w,b):
    m = X.shape[0] #No. of Rows of X
    y_pred = np.dot(X,w) + b*np.ones([m,1]) #Predicted y
    error = (1/(2*m))*np.dot((y_pred - y).T,(y_pred - y)) #Error Calculation
    return error

#Function to calculate the loss
def CalcLoss(X,y,w,b,lambdav):
    m = X.shape[0] #No. of Rows of X
    y_pred = np.dot(X,w) + b*np.ones([m,1]) #Predicted y
    loss = (1/(2*m))*np.dot((y_pred - y).T,(y_pred - y)) + lambdav*np.dot(w.T,w)#Loss Calculation
    return loss

#Function to calculate the gradient descent ridge regression error and loss for training and testing data
def ridge_regression_gradient_descent(X, y, testX, testY, lambdav, learning_rate, num_iterations,tol):
    start = time.time() #Start timer to get runtime
    m, n = X.shape
    w = np.zeros(n).reshape(-1,1)  # Initialize coefficients to zeros
    b = 0
    one = np.ones([m,1])
    train_loss_list = np.array([])
    it = np.array([])
    for iter in range(num_iterations):
        #A list of training losses for every iteration before hitting tolerance
        train_loss_list = np.append(train_loss_list,CalcLoss(X,y,w,b,lambdav))
        it = np.append(it,iter) #A list of iterations
        # Calculate the predicted values
        y_pred = np.dot(X, w).reshape(-1,1) + b*one  
        # Calculate the gradient of the loss function with respect to w
        wgradient = (1/m) * np.dot(X.T, y_pred - y) + 2 * lambdav * w
        # Calculate the gradient of the loss function with respect to b
        bgradient = (1/m) * np.dot(one.T, y_pred - y)
        w_t = w.copy()
        # Update the coefficients using gradient descent
        w -= learning_rate * wgradient
        b -= learning_rate * bgradient
        if(np. linalg. norm(w - w_t)  <= tol): #Checking difference of weights to tolerance for exit
            break        
    end = time.time() #Stop timer and calculate runtime
    runtime = end - start
    train_error = CalcError(X,y,w,b) #Get the training error of input data
    train_loss = CalcLoss(X,y,w,b,lambdav) #Get the training loss of input data
    test_error = CalcError(testX,testY,w,b) #Get the testing error of input data
    return train_error, train_loss, test_error, runtime, train_loss_list, it

#Function to calculate training and testing error and loss for Closed form regression
def ridge_regression_closed_form(X, y, testX, testY, lambdav):
    start = time.time()      #Counter for runtime
    Xt = np.hstack((X, np.ones([X.shape[0],1]))) #Add a column of 1s at the end of X
    A = np.dot(Xt.T, Xt) + lambdav * 2*np.eye(Xt.shape[1]) *len(y) #Derived calculation for A
    # Calculate the closed-form solution for ridge regression
    w = np.linalg.solve(A, np.dot(Xt.T, y)) #Solve to get coefficients
    b = w[-1,:] #Seperate weights from coefficients
    w = w[:-1,:]#Seperate bias from coefficients
    train_error = CalcError(X,y,w,b)
    train_loss = CalcLoss(X,y,w,b,lambdav)
    test_error = CalcError(testX,testY,w,b)
    end = time.time()  #Stopping runtime counter
    runtime = end - start
    return train_error, train_loss, test_error, runtime

#A Function to Read and standardize data before using them on models
def data_prep():
    #Read data from the given files and format them
    X_train = np.genfromtxt('a1-files/housing_X_train.csv', delimiter=',')
    X_test = np.genfromtxt('a1-files/housing_X_test.csv', delimiter=',')
    Y_train = np.genfromtxt('a1-files/housing_Y_train.csv', delimiter=',').reshape(-1,1) 
    Y_test = np.genfromtxt('a1-files/housing_Y_test.csv', delimiter=',').reshape(-1,1)
    X_train = X_train.T #Transpose X_matrix to make them useable
    X_test = X_test.T
    #Standardize all read data
    X_train = (X_train - np.mean(X_train, axis=0))/np.std(X_train, axis=0)
    X_test = (X_test - np.mean(X_test, axis=0))/np.std(X_test, axis=0)
    Y_train = (Y_train - np.mean(Y_train, axis=0))/np.std(Y_train, axis=0)
    Y_test = (Y_test - np.mean(Y_test, axis=0))/np.std(Y_test, axis=0)
    return X_train, X_test, Y_train, Y_test 
    
def control(learning_rate = 0.0001, num_iterations = 100000000,tol = 1e-6):
    #Accept X and Y data
    X_train, X_test, Y_train, Y_test = data_prep()
    #Initializing all lists to store return values from functions
    train_error = [1,1,1,1]
    train_loss = [1,1,1,1]
    test_error = [1,1,1,1]
    runtime = [1,1,1,1]
    train_loss_list =[1,1]
    it = [1,1]
    
    #Ridge Regression Calculations
    #Closed form for lambda = 0
    lambdav = 0
    train_error[0], train_loss[0], test_error[0], runtime[0] = ridge_regression_closed_form(X_train, Y_train, X_test, Y_test, lambdav)
    #Gradient Descent for lambda =0
    train_error[2], train_loss[2], test_error[2], runtime[2], train_loss_list[0], it[0] = ridge_regression_gradient_descent(X_train, Y_train, X_test, Y_test, lambdav, learning_rate, num_iterations,tol)
    #Closed form for lambda =10
    lambdav = 10
    train_error[1], train_loss[1], test_error[1], runtime[1] = ridge_regression_closed_form(X_train, Y_train, X_test, Y_test, lambdav)
    #Gradient Descent for lambda =10
    train_error[3], train_loss[3], test_error[3], runtime[3], train_loss_list[1],it[1] = ridge_regression_gradient_descent(X_train, Y_train, X_test, Y_test, lambdav, learning_rate, num_iterations,tol)
    
    #Print all training error, and training loss of training data
    print("order of output")
    print("[lambda=0 closed form, lambda=10 closed form,lambda=0 gradient descent,lambda=10 gradient descent]")
    print("training error",train_error)
    print("training loss",train_loss)
    print("testing error", test_error) #testing error of testing data
    print(f"Runtime of lambda 0 closed form: {runtime[0]:.20f}")
    print(f"Runtime of lambda 10 closed form: {runtime[1]:.20f}")
    print(f"Runtime of lambda 0 Gradient Descent: {runtime[2]:.5f}")
    print(f"Runtime of lambda 10 Gradient Descent: {runtime[3]:.5f}")

    #Plot all required graphs
    plt.scatter(it[0],train_loss_list[0],color='green',label='lambda = 0')
    #Plot training loss vs iteration for lambda 0 and 10 for Gradient Descent
    plt.xlabel('Iteration Number')
    plt.ylabel('Training Loss')
    plt.title('Training Loss vs Iteration Number for lambda = 0')
    plt.show()
    plt.scatter(it[1],train_loss_list[1],color='red',label='lambda = 10')
    plt.xlabel('Iteration Number')
    plt.ylabel('Training Loss')
    plt.title('Training Loss vs Iteration Number for lambda = 10')
    plt.show()

#Function call to run the program
control()