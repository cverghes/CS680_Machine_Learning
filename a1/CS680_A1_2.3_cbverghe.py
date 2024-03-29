
############################################################################################
#Course: CS680
#Name: Chris Binoi Verghese
#Course: MDSAI
#Description: Implements ridge regression using the closed form solution for linear regression
############################################################################################

import numpy as np

#Function to accept X,y and lambda value to calculate ridge regression using the closed form
def ridge_regression_closed_form(X, y, lambdav):
    Xt = np.hstack((X, np.ones([X.shape[0],1]))) #Add a column of 1s at the end of X
    A = np.dot(Xt.T, Xt) + lambdav * 2*np.eye(Xt.shape[1]) *len(y) #Derived calculation for A
    # Calculate the closed-form solution for ridge regression
    w = np.linalg.solve(A, np.dot(Xt.T, y)) #Solve to get coefficients
    b = w[-1,:] #Seperate weights from coefficients
    w = w[:-1,:]#Seperate bias from coefficients
    return w,b
