
############################################################################################
#Course: CS680
#Name: Chris Binoi Verghese
#Course: MDSAI
#Assignmnent 2 Question 1 
#Description: Implements Logistic Regression, Softmargin and Hard Margin SVM on  datasets and learning about them
############################################################################################

import numpy as np
import statsmodels.api as sm 
import matplotlib.pyplot as plt
import sklearn.svm as svm

#Function meant to read data from required files
def data_prep(file_name1 , file_name2):
    #Read data from the given files and format them
    X_train = np.genfromtxt(file_name1, delimiter=',')
    Y_train = np.genfromtxt(file_name2, delimiter=',').reshape(-1)      
    #Standardize all read X data
    X_train = (X_train - np.mean(X_train, axis=0))/np.std(X_train, axis=0)
    return X_train, Y_train

#Function answering prompts from Q1.1 involving training data A
def model_predict11(X_train, Y_train):
    #L(y, p )=−[ylog(p^ )+(1−y)log(1−p^ )] - The Loss Function of Logistic regression
    #Needs a regularized logistic regression
    log_reg = sm.Logit(Y_train, X_train).fit_regularized()
    #The soft margin SVM with c = 1
    clf = svm.SVC(C = 1,kernel="linear").fit( X_train,Y_train)
    #The hard margin SVM with c = float(inf)
    clf2 = svm.SVC(C = float('inf'),kernel="linear").fit( X_train,Y_train)
    return clf, clf2,log_reg

#Function answering prompts from Q1.2
def model_predict12(X_train, Y_train):
    #This dataset does not require regularized logistic regression
    log_reg = sm.Logit(Y_train, X_train).fit()
    #The soft margin SVM with c = 1
    clf = svm.SVC(C = 1,kernel="linear").fit( X_train,Y_train)
    #The hard margin SVM with c = float(inf) that requires a max iteration for this dataset
    clf2 = svm.SVC(C = float('inf'),kernel="linear", max_iter= 1000).fit( X_train,Y_train)
    return clf, clf2,log_reg

#Function to calculate parameter vectors, support vectors and their required points
def check(soft_svm, X_train, Y_train):
    coef = soft_svm.coef_ # Coefficient vector
    pred = np.dot(X_train, coef.T)
    c = 0
    pred = np.round(pred,0)
    for i in pred:
        if i<=1:
            c += 1
    print(f"Number of points <= 1: ",c)
    #Obtaining support vectors and dual coefs to calculate the parameter vector
    support_vectors = soft_svm.support_vectors_
    dual_coef = soft_svm.dual_coef_
    parameter_vector = np.dot(dual_coef,support_vectors)
    return parameter_vector, len(support_vectors)


#Read all required data
X_train_A,Y_train_A = data_prep('a2-files\data\X_train_A.csv', 'a2-files\data\Y_train_A.csv')
X_train_B,Y_train_B = data_prep('a2-files\data\X_train_B.csv', 'a2-files\data\Y_train_B.csv')
X_test_B,Y_test_B = data_prep('a2-files\data\X_test_B.csv', 'a2-files\data\Y_test_B.csv')

soft,hard,log = model_predict11(X_train_A,Y_train_A)
print(f"Parameter vector followed by number of required points:\n",check(soft, X_train_A, Y_train_A))
soft,hard,log = model_predict12(X_train_B,Y_train_B)
print(f"Parameter vector followed by number of required points:\n",check(soft, X_train_B, Y_train_B))
svm_accuracy = np.mean(soft.predict(X_test_B) == Y_test_B)
print(f"The 0-1 Loss of the soft-margin SVM model: ",svm_accuracy)
svm_accuracy = np.mean(hard.predict(X_test_B) == Y_test_B)
print(f"The 0-1 Loss of the hard-margin SVM model: ",svm_accuracy)
#Seperating all y values into 0s and 1s
l_attributed = [1 if v > 0.01 else 0 for v in log.predict(X_test_B)]
svm_accuracy = np.mean(l_attributed == Y_test_B)
print(f"The 0-1 Loss of the logistic regression model: ",svm_accuracy)
