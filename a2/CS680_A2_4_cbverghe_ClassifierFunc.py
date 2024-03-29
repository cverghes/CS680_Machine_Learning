
############################################################################################
#Course: CS680
#Name: Chris Binoi Verghese
#Course: MDSAI
#Assignmnent 2 Question 4
#Description: Implements Decision Tree, hoever it uses Decision TreeClassifier
############################################################################################

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.metrics import accuracy_score

class DecisionTree:

    def __init__(self, criterion='gini'):
        self.criterion = criterion
        self.train_accuracy = []
        self.test_accuracy= []
        self.tree = None
    #Build the decision Tree
    def build(self, X, y,max_depth):
        self.tree = DecisionTreeClassifier(max_depth=max_depth, criterion=self.criterion)
        self.tree.fit(X, y)
        
    #Predict y data and calculate their accuracy
    def predict(self, X_train,X_test):
        if self.tree is not None:
            y_train_pred = self.tree.predict(X_train)
            y_test_pred = self.tree.predict(X_test)
            self.train_accuracy.append(accuracy_score(y_train, y_train_pred))
            self.test_accuracy.append(accuracy_score(y_test, y_test_pred))

    def get_train(self):
        return self.train_accuracy
    
    def get_test(self):
        return self.test_accuracy
        
#Read all required data
X_train = pd.read_csv('a2-files\data\X_train_D.csv')
y_train = pd.read_csv('a2-files\data\y_train_D.csv')
X_test = pd.read_csv('a2-files\data\X_test_D.csv')
y_test = pd.read_csv('a2-files\data\y_test_D.csv')
X_test.columns = X_train.columns


# Define the maximum depth range starting from 1
max_depth_range = range(1, 20)
tree_misclass = DecisionTree(criterion='gini')
tree_gini = DecisionTree(criterion='gini')
tree_entropy = DecisionTree(criterion='entropy')

# Loop over different maximum depths
for max_depth in max_depth_range:
    tree_misclass.build(X_train, y_train, max_depth)
    tree_misclass.predict(X_train, X_test)
    tree_gini.build(X_train, y_train, max_depth)
    tree_gini.predict(X_train, X_test)
    tree_entropy.build(X_train, y_train, max_depth)
    tree_entropy.predict(X_train, X_test)


# Create plots for accuracy vs. maximum depth
plt.figure(figsize=(12, 6))

# Misclassification error plot
plt.subplot(321)
plt.plot(max_depth_range, tree_misclass.get_train(), label='Train Accuracy')
plt.plot(max_depth_range, tree_misclass.get_test(), label='Test Accuracy')
plt.xlabel('Maximum Depth')
plt.ylabel('Accuracy')
plt.title('Misclassification Error')
plt.legend()

# Gini index plot
plt.subplot(322)
plt.plot(max_depth_range, tree_gini.get_train(), label='Train Accuracy')
plt.plot(max_depth_range, tree_gini.get_test(), label='Test Accuracy')
plt.xlabel('Maximum Depth')
plt.ylabel('Accuracy')
plt.title('Gini Index')
plt.legend()

# Entropy plot
plt.subplot(323)
plt.plot(max_depth_range, tree_entropy.get_train(), label='Train Accuracy')
plt.plot(max_depth_range, tree_entropy.get_test(), label='Test Accuracy')
plt.xlabel('Maximum Depth')
plt.ylabel('Accuracy')
plt.title('Entropy')
plt.legend()

plt.tight_layout()
plt.show()
