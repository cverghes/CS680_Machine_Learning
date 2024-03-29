
############################################################################################
#Course: CS680
#Name: Chris Binoi Verghese
#Course: MDSAI
#Description: Implements the gradient descent algorithm for solving ridge regression.
############################################################################################

import numpy as np

#Function to implement gradient descent algorithm
def ridge_regression_gradient_descent(X, y, lambdav = 5, learning_rate = 0.000001, num_iterations  = 1000000,tol = 1e-6):
    m, n = X.shape       
    w = np.zeros(n).reshape(-1,1)  # Initialize coefficients to zeros
    b = 0
    one = np.ones([m,1])
    for iteration in range(num_iterations):
        # Calculate the predicted values
        y_pred = np.dot(X, w).reshape(-1,1) + b*one
        # Calculate the gradient of the loss function with respect to w
        wgradient = (1/m) * np.dot(X.T, y_pred - y) + 2 * lambdav * w
        # Calculate the gradient of the loss function with respect to b
        bgradient = (1/m) * np.dot(one.T, y_pred - y)
        w_t = w.copy()  #Copy the previous weights before update
        # Update the coefficients using gradient descent for eights and bias
        w -= learning_rate * wgradient
        b -= learning_rate * bgradient
        if(np. linalg. norm(w - w_t)  <= tol): #Check ith tolerance to exit
            break
    return w,b


