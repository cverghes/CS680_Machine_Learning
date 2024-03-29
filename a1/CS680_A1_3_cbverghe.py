
############################################################################################
#Course: CS680
#Name: Chris Binoi Verghese
#Course: MDSAI
#Description: Trains linear regression, ridge regression, and lasso on three mystery datasets
############################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error

# Function to load datasets from csv files
def load_data(prefix):
    X_train = pd.read_csv(f'a1-files/X_train_{prefix}.csv', header=None)
    Y_train = pd.read_csv(f'a1-files/Y_train_{prefix}.csv', header=None)
    X_test = pd.read_csv(f'a1-files/X_test_{prefix}.csv', header=None)
    Y_test = pd.read_csv(f'a1-files/Y_test_{prefix}.csv', header=None)
    return X_train, Y_train, X_test, Y_test

# Function to print MSE and parameters for the three regressions
def print_output(mse_lin,alpha_ridge, alpha_lasso, mse_ridge, mse_lasso,data):
    print(f"Linear Regression for {data} MSE: {mse_lin:.4f}")
    print(f"Ridge Regression for {data} (alpha={alpha_ridge}) MSE: {mse_ridge:.4f}")
    print(f"Lasso Regression for {data} (alpha={alpha_lasso}) MSE: {mse_lasso:.4f}\n")
    return

# Creates Histograms for parameter vector for all three datasets
def plot_histo( linear, ridge,lasso, title):
    #A histogram to show all three regression approaches
    par = np.concatenate((linear, ridge,lasso))
    plt.hist(par.T, bins=25, label=['Linear Regression', 'Ridge Regression', 'Lasso Regression'], color=['red', 'blue', 'green'])
    plt.xlabel('Values of coordinates')
    plt.ylabel('Number of coordinates')
    plt.title(title)
    plt.legend()
    plt.show()


# Function for Ridge and Lasso regression with Cross-Validation
def kfold(X, Y, alpha, model):
    #Set initial values for all variables
    alphabest = -1
    bestmse = float('inf')
    k = 6
    fold = len(X) // k
    indices = np.arange(len(X))
    #Run a loop to cycle through all values of alpha to find the best one
    for a in alpha:
        mse_val = [] #Holds list of mse values for all the data folds
        for i in range(k):
            #Ready all data by converting them into their respective fold before training the model
            val_index = indices[i * fold:(i + 1) * fold]
            train_index = np.concatenate((indices[:i * fold], indices[(i + 1) * fold:]))
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            Y_train, y_val = Y.iloc[train_index], Y.iloc[val_index]
            #Choose model mased on input parameter
            if model == 'R':
                reg = Ridge(a)
            elif model == 'L':
                reg = Lasso(a)
            y_pred = reg.fit(X_train, Y_train).predict(X_val) #Predicted Y from model
            mse_val.append(mean_squared_error(y_val, y_pred)) #Calculates MSE of model
        #Find the mean MSE of all k folds
        mse = np.mean(mse_val)
        #Assign ne best alpha if there is a better MSE
        if mse < bestmse:
            bestmse = mse
            alphabest = a

    return alphabest

#Initialize all variables and lists used
mse_lin = [1,1,1]
mse_ridge = [1,1,1]
mse_lasso = [1,1,1]
# We chose these alpha values as they cover a range of magnitudes, starting from very small to moderately large values.
alpha = [0.001, 0.01, 0.1, 1.0, 10.0]
alphabest = [1,1,1,1,1,1]

# Load datasets for A, B, and C from their excel files
X_train_A, Y_train_A, X_test_A, y_test_A = load_data('A')
X_train_B, Y_train_B, X_test_B, y_test_B = load_data('B')
X_train_C, Y_train_C, X_test_C, y_test_C = load_data('C')

# Trainlinear regression models for the three datasets
lin_A = LinearRegression().fit(X_train_A, Y_train_A)
lin_B = LinearRegression().fit(X_train_B, Y_train_B)
lin_C = LinearRegression().fit(X_train_C, Y_train_C)


# Train Ridge models using kfold() to get the best alpha model and then Ridge() to train a model
alphabest[0]= kfold(X_train_A, Y_train_A, alpha, 'R')
ridge_A = Ridge(alphabest[0]).fit(X_train_A, Y_train_A)
alphabest[1] = kfold(X_train_B, Y_train_B, alpha, 'R')
ridge_B = Ridge(alphabest[1]).fit(X_train_B, Y_train_B)
alphabest[2] = kfold(X_train_C, Y_train_C, alpha, 'R')
ridge_C = Ridge(alphabest[2]).fit(X_train_C, Y_train_C)

# Train Lasso models using kfold() to get the best alpha model and then Lasso() to train a model
alphabest[3]= kfold(X_train_A, Y_train_A, alpha, 'L')
lasso_A = Lasso(alphabest[3]).fit(X_train_A, Y_train_A)
alphabest[4] = kfold(X_train_B, Y_train_B, alpha, 'L')
lasso_B = Lasso(alphabest[4]).fit(X_train_B, Y_train_B)
alphabest[5]= kfold(X_train_C, Y_train_C, alpha, 'L')
lasso_C = Lasso(alphabest[5]).fit(X_train_C, Y_train_C)

# Calculate Mean Squared Errors for test datasets for all three regressions
mse_lin[0] = mean_squared_error(y_test_A, lin_A.predict(X_test_A))
mse_ridge[0] = mean_squared_error(y_test_A, ridge_A.predict(X_test_A))
mse_lasso[0] = mean_squared_error(y_test_A, lasso_A.predict(X_test_A))
mse_lin[1] = mean_squared_error(y_test_B, lin_B.predict(X_test_B))
mse_ridge[1] = mean_squared_error(y_test_B, ridge_B.predict(X_test_B))
mse_lasso[1] = mean_squared_error(y_test_B, lasso_B.predict(X_test_B))
mse_lin[2] = mean_squared_error(y_test_C, lin_C.predict(X_test_C))
mse_ridge[2] = mean_squared_error(y_test_C, ridge_C.predict(X_test_C))
mse_lasso[2] = mean_squared_error(y_test_C, lasso_C.predict(X_test_C))

# Call the function to print the MSE and parameters for the three regressions
print_output(mse_lin[0],alphabest[0], alphabest[3], mse_ridge[0], mse_lasso[0],'A')
print_output(mse_lin[1],alphabest[1], alphabest[4], mse_ridge[1], mse_lasso[1],'B')
print_output(mse_lin[2],alphabest[2], alphabest[5], mse_ridge[2], mse_lasso[2],'C')

#Call the function to set up histograms to display the coefficients of the three models
plot_histo(lin_A.coef_.reshape(1, -1), ridge_A.coef_.reshape(1, -1), lasso_A.coef_.reshape(1, -1), 'Parameters for A')
plot_histo(lin_B.coef_.reshape(1, -1), ridge_B.coef_.reshape(1, -1), lasso_B.coef_.reshape(1, -1), 'Parameters for B')
plot_histo(lin_C.coef_.reshape(1, -1), ridge_C.coef_.reshape(1, -1), lasso_C.coef_.reshape(1, -1), 'Parameters for C')

