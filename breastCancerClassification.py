# importing data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
# print(cancer)

# get keys within dictionaries
cancer['DESCR']
cancer['target']
cancer['target_names']
cancer['feature_names']
cancer['data']

df_cancer = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns=np.append(cancer['feature_names'], ['target']))
print(df_cancer)

# visualizing data
sns.pairplot(df_cancer,
             hue = 'target',
             vars=['mean radius', 'mean texture', 'mean area', 'mean perimeter', 'mean smoothness'])

sns.countplot(df_cancer['target'])
sns.scatterplot(x = 'mean area', y = 'mean smoothness', hue = 'target', data = df_cancer)

plt.figure(figsize = (20, 10))
sns.heatmap(df_cancer.corr(), annot = True)

# model training, finding a problem solution
# drop target column
X = df_cancer.drop(['target'], axis=1)
y = df_cancer['target']

# split dataset into training & test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.2, random_state=5)

# classificationw with SVM
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
svc_model = SVC()
svc_model.fit(X_train, y_train)

# model evaluation
y_predict = svc_model.predict(X_test) # just 1s so useless
# y_test true target value, y_predict the models' predictions
cm = confusion_matrix(y_test, y_predict) # horrible result

'''
optimizing the models' performance
C param: controls trade-off between classifying training points correctly
          & having a smooth decision boundary

Small C (loose): makes cost penalty or misclassification low (soft margin)

Large C (strict): makes cost of misclassification high (hard margin), forcing
                 the model to explain input data stricter & potentially over fit

Gamma Param: controls how far the influence of a single training set reaches
Large Gamma: close reach, closer data points have high weight
Small Gamma: far reach, more generalized solution)
'''

# improving model
min_train = X_train.min()
range_train = (X_train - min_train).max()
X_train_scaled = (X_train - min_train) / range_train

# shows training data without scaling
sns.scatterplot(x = X_train['mean area'], y = X_train['mean smoothness'], hue = y_train)

# shows training data with scaling
sns.scatterplot(x = X_train_scaled['mean area'], y = X_train_scaled['mean smoothness'], hue = y_train)

# normalization on test set !!!
min_test = X_test.min()
range_test = (X_test - min_test).max()
X_test_scaled = (X_test - min_test) / range_test

svc_model.fit(X_train_scaled, y_train)
y_predict = svc_model.predict(X_test_scaled)

cm = confusion_matrix(y_test, y_predict)

sns.heatmap(cm, annot = True)

# classification report
print(classification_report(y_test, y_predict))

# optimize C & gamma params
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']}

from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 4)
grid.fit(X_train_scaled, y_train)

grid.best_params_
grid_predictions = grid.predict(X_test_scaled)
cm = confusion_matrix(y_test, grid_predictions)
sns.heatmap(cm, annot = True)

print(classification_report(y_test, y_predict))















