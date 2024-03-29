
############################################################################################
#Course: CS680
#Name: Chris Binoi Verghese
#Course: MDSAI
#Assignmnent 2 Question 4
#Description: Implements Decision Tree Classifier for varying depths
############################################################################################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#The tree node
class TreeNode:
    def __init__(self, depth):
        self.depth = depth
        self.feature_index = None
        self.threshold = None
        self.left = None
        self.right = None
        self.value = None

#The Decision tree class
class DecisionTree:
    def __init__(self, max_depth=None, criterion='gini'):
        self.max_depth = max_depth
        self.criterion = criterion
        self.tree = None

    #Code to build the tree hen data is being fit in
    def build(self, X, y, depth=0):
        #If max depth is already reached
        if depth == self.max_depth or len(np.unique(y)) == 1:
            node = TreeNode(depth)
            node.value = y.astype(int)
            return node

        best_criterion = float('inf')
        best_split = None
        best_left_X, best_left_y, best_right_X, best_right_y = None, None, None, None

        for i in range(X.shape[1]):
            unique_values = np.unique(X[:, i])
            unique_values.sort()

            for j in range(len(unique_values) - 1):
                threshold = (unique_values[j] + unique_values[j + 1]) / 2
                left_indices = X[:, i] <= threshold
                right_indices = X[:, i] > threshold

                if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
                    continue

                left_y = y[left_indices]
                right_y = y[right_indices]

                if self.criterion == 'gini':
                    loss = gini_coefficient(left_y, right_y)
                elif self.criterion == 'entropy':
                    loss = entropy(left_y, right_y)
                elif self.criterion == 'misclassification':
                    loss = misclassification_error(left_y, right_y)

                if loss < best_criterion:
                    best_criterion = loss
                    best_split = (i, threshold)
                    best_left_X = X[left_indices]
                    best_left_y = y[left_indices]
                    best_right_X = X[right_indices]
                    best_right_y = y[right_indices]

        if best_criterion == float('inf'):
            node = TreeNode(depth)
            node.value = np.argmax(np.bincount(y.squeeze().astype(int)))
            return node

        node = TreeNode(depth)
        node.feature_index, node.threshold = self.find_best_split(X, y)
        node.left = self.build(best_left_X, best_left_y, depth + 1)
        node.right = self.build(best_right_X, best_right_y, depth + 1)

        return node

    #Finiding the best split for the tree to get best feature and threshold
    def find_best_split(self,X, y):
        best_loss = float('inf')
        best_feature = None
        best_threshold = None

        for feature_index in range(X.shape[1]):
            unique_values = np.unique(X[:, feature_index])
            unique_values.sort()

            for j in range(len(unique_values) - 1):
                threshold = (unique_values[j] + unique_values[j + 1]) / 2

                left_indices = X[:, feature_index] <= threshold
                right_indices = X[:, feature_index] > threshold

                SL = y[left_indices]
                SR = y[right_indices]
                #Choosing the lose based the chosen criterion
                if self.criterion == 'gini':
                    loss = len(SL) * gini_coefficient(SL,SR)
                elif self.criterion == 'entropy':
                    loss = len(SL) * entropy(SL,SR)
                elif self.criterion == 'misclassification':
                    loss = len(SL) * misclassification_error(SL,SR)
                    #Choosing the least loss
                if loss < best_loss:
                    best_loss = loss
                    best_feature = feature_index
                    best_threshold = threshold

        return best_feature, best_threshold
    
    #Fitting the model in order to build it
    def fit(self, X, y):
            self.tree = self.build(X, y)

    #A resursive prediction function to keep going till the threshold is reached
    def predict_recurse(self, X, node):
            if node.value is not None:
                return node.value
            if X[node.feature_index] <= node.threshold:
                return self.predict_recurse(X, node.left)
            else:
                return self.predict_recurse(X, node.right)
            
    #Function to start processing predictions
    def predict(self, X):
            predictions = [self.predict_recurse(x, self.tree) for x in X]
            return predictions

#Function to calculate the Misclassification error from y right and y left
def misclassification_error(yL, yR):
    pL = np.mean(yL)
    pR = np.mean(yR)
    return min(pL, 1 - pL) + min(pR, 1 - pR)

#Function to calculate the gini coefficient from y right and y left
def gini_coefficient(yL, yR):
    pL = np.mean(yL)
    pR = np.mean(yR)
    return pL * (1 - pL) + pR * (1 - pR)

#Function to calculate the entropy loss from y right and y left
def entropy(yL, yR):
    pL = np.mean(yL)
    pR = np.mean(yR)
    if pL == 0 or pL == 1:
        entropyL = 0
    else:
        entropyL = -pL * np.log2(pL) - (1 - pL) * np.log2(1 - pL)

    if pR == 0 or pR == 1:
        entropyR = 0
    else:
        entropyR = -pR * np.log2(pR) - (1 - pR) * np.log2(1 - pR)

    return pL * entropyL + pR * entropyR

#Read the data from dataset D
X_train = pd.read_csv('a2-files/data/X_train_D.csv').values
y_train = pd.read_csv('a2-files/data/y_train_D.csv').values
X_test = pd.read_csv('a2-files/data/X_test_D.csv').values
y_test = pd.read_csv('a2-files/data/y_test_D.csv').values

train_accuracy = []
test_accuracy = []

max_depth_range = range(1, 20)

#Calculate the accuracy from traing and testing data for Misclassification loss and plot it
for max_depth in max_depth_range:
    tree = DecisionTree(max_depth, criterion='misclassification')
    tree.fit(X_train, y_train)
    y_train_pred = tree.predict(X_train)
    y_test_pred = tree.predict(X_test)
    y_train_pred = np.array(y_train_pred)
    list = []
    for i in y_train_pred[0]:
        list.append(i[0])
  
    train_accuracy.append(np.mean(y_train == list))
    list = []
    for i in y_test_pred[0]:
        list.append(i[0])
    test_accuracy.append(np.mean(y_test == list))

plt.figure(figsize=(12, 6))
plt.subplot(321)
plt.plot(max_depth_range, train_accuracy, label='Train Accuracy')
plt.plot(max_depth_range, test_accuracy, label='Test Accuracy')
plt.xlabel('Maximum Depth')
plt.ylabel('Accuracy')
plt.title('Misclassification Error')
plt.legend()

#Calculate the accuracy from traing and testing data for Gini Index loss and plot it
train_accuracy = []
test_accuracy = []
for max_depth in max_depth_range:
    tree = DecisionTree(max_depth, criterion='gini')
    tree.fit(X_train, y_train)
    y_train_pred = tree.predict(X_train)
    y_test_pred = tree.predict(X_test)
    y_train_pred = np.array(y_train_pred)
    list = []
    for i in y_train_pred[0]:
        list.append(i[0])
  
    train_accuracy.append(np.mean(y_train == list))
    list = []
    for i in y_test_pred[0]:
        list.append(i[0])
    test_accuracy.append(np.mean(y_test == list))

plt.subplot(322)
plt.plot(max_depth_range, train_accuracy, label='Train Accuracy')
plt.plot(max_depth_range, test_accuracy, label='Test Accuracy')
plt.xlabel('Maximum Depth')
plt.ylabel('Accuracy')
plt.title('Gini Index')
plt.legend()

#Calculate the accuracy from traing and testing data for entropy loss and plot it
train_accuracy = []
test_accuracy = []
for max_depth in max_depth_range:
    tree = DecisionTree(max_depth, criterion='entropy')
    tree.fit(X_train, y_train)
    y_train_pred = tree.predict(X_train)
    y_test_pred = tree.predict(X_test)
    y_train_pred = np.array(y_train_pred)
    list = []
    for i in y_train_pred[0]:
        list.append(i[0])
  
    train_accuracy.append(np.mean(y_train == list))
    list = []
    for i in y_test_pred[0]:
        list.append(i[0])
    test_accuracy.append(np.mean(y_test == list))

plt.subplot(323)
plt.plot(max_depth_range, train_accuracy, label='Train Accuracy')
plt.plot(max_depth_range, test_accuracy, label='Test Accuracy')
plt.xlabel('Maximum Depth')
plt.ylabel('Accuracy')
plt.title('Entropy')
plt.legend()

plt.tight_layout()
plt.show()