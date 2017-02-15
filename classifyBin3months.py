#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
# project: Insight-Churn
# date: 25/01/2017
# author: Deniz Ustebay
# description: Run Logistic Regression Classifier with CV
               get accuracy
"""

from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import psycopg2
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import os
from collections import defaultdict
import time
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import seaborn as sns
import statsmodels.api as sm
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import itertools

os.chdir('/Users/deniz/Research/Insight_Churn/')
plt.style.use('ggplot')

# %%
data_train = pd.read_pickle('dataset_forLogisticRegression_TRAIN.pkl') 
data_test = pd.read_pickle('dataset_forLogisticRegression_TEST.pkl') 



print data_train.shape
print data_test.shape
data_train.head()


# %%
features_cols = ['id','churned_in_target', 'city_id','age','years_of_experience',\
                 'total_holidays_initial','average_client_score','occupation_ratio',\
                 'double_shift_ratio','office_shifts_ratio']


# keep columns we are interested in
features_train = data_train[features_cols]
features_test = data_test[features_cols]

features_train.head()


# ##

# Scale and adjust dummy variables here 
train_cols = features_cols[3:]
X_train = features_train[train_cols]
# city is caegorical: one hot decoding
# 3 categories yield two dummy features (third one is obvious from the two)
dummy_city = pd.get_dummies(features_train['city_id'], prefix='city_id')
X_train = X_train.join(dummy_city.ix[:, 'city_id_2':])
X_train["intercept"] = 1.0
for c in train_cols:
    X_train[c] = (X_train[c]-X_train[c].mean())/X_train[c].std()
    
y_train = features_train['churned_in_target']


X_test = features_test[train_cols]
dummy_city = pd.get_dummies(features_test['city_id'], prefix='city_id')
X_test = X_test.join(dummy_city.ix[:, 'city_id_2':])
X_test["intercept"] = 1.0
for c in train_cols:
    X_test[c] = (X_test[c]-X_test[c].mean())/X_test[c].std()
    
y_test = features_test['churned_in_target']


# %%
fig = plt.figure(figsize=(8,6))
# Compute the correlation matrix
corr = X_train[X_train.columns[range(len(X_train.columns)-1)]].corr()
corr.index = ['Age','Prior experience','Time-off','Client score','Workload','Double shifts','Work type','City A','City B']
corr.columns = ['Age','Prior experience','Time-off','Client score','Workload','Double shifts','Work type','City A','City B']

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, cmap=cmap, vmax=.3,square=True,
            linewidths=.5, cbar_kws={"shrink": .5})
plt.yticks(rotation=0) 
plt.xticks(rotation=90) 
plt.tick_params(axis='both', which='major', labelsize=16)
cax = plt.gcf().axes[-1]
cax.tick_params(labelsize=16)
plt.tight_layout()
plt.savefig('correlations.png') 


# %% 
# Scatter plot matrix of features
fig = plt.figure(figsize=(16,10))
sns.pairplot(features_train[train_cols[2:-1]+['churned_in_target']], hue="churned_in_target")
plt.savefig('scatterplot_matrix.png')


# cross validation to choose C (regularization parameter)
number_folds = 10
lambda_power = np.linspace(-2,2,num=10)
C_list =  np.zeros(len(lambda_power))
mean_scores = np.zeros(len(lambda_power))
std_scores = np.zeros(len(lambda_power))
for i in range(len(lambda_power)):
    C_list[i] = 1.0/(10**lambda_power[i])
    classifier = LogisticRegression(C=C_list[i], penalty='l2')
    scores = cross_val_score(classifier, X_train, y_train, cv=number_folds)
    mean_scores[i] = scores.mean()
    std_scores[i] = scores.std()

plt.figure()
plt.semilogx(1.0/C_list, mean_scores, marker='o',lw= 3,label='l2')

C_list =  np.zeros(len(lambda_power))
mean_scores = np.zeros(len(lambda_power))
std_scores = np.zeros(len(lambda_power))
for i in range(len(lambda_power)):
    C_list[i] = 1.0/(10**lambda_power[i])
    classifier = LogisticRegression(C=C_list[i], penalty='l1')
    scores = cross_val_score(classifier, X_train, y_train, cv=number_folds)
    mean_scores[i] = scores.mean()
    std_scores[i] = scores.std()

plt.semilogx(1.0/C_list, mean_scores, marker='o',lw= 3,label='l1')
plt.ylabel('Acccuracy')
plt.xlabel('Regularization parameter')
plt.legend()


# %%

# Logistic regression regularization parameter (C = 1/lambda)
C1 = 1.0/7 # L1 
C2 = 1.0/5 # L2
# Create different classifiers:
classifiers = {
               'L1 logistic': LogisticRegression(C=C1, penalty='l1'),
               'L2 logistic (OvR)': LogisticRegression(C=C2, penalty='l2')
               }

n_classifiers = len(classifiers)

for index, (name, classifier) in enumerate(classifiers.items()):
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    prediction_accuracy = np.mean(y_pred.ravel() == y_test.ravel()) * 100
    scores = cross_val_score(classifier, X_train, y_train, cv=10)
    print("mean cross validation score : %f " % (np.mean(scores)))
    print("CV accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print("prediction accuracy for %s : %f " % (name, prediction_accuracy))
    print classifier.coef_
    print ''
    
        
    preds = classifier.predict_proba(X_test)[:,1]
    fpr, tpr, _ = metrics.roc_curve(y_test, preds)
    
    auc = metrics.auc(fpr,tpr)
    
    fig = plt.figure(figsize=(6,6))
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',lw=lw)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate',fontsize=16)
    plt.ylabel('True Positive Rate',fontsize=16)
    plt.title("ROC Curve w/ AUC= %0.2f" % auc,fontsize=16)
    plt.tight_layout()
    
    plt.legend(loc="lower right")
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.savefig('ROC.png') 


# Statmodels
logit = sm.Logit(y_train,X_train)
# fit the model
result = logit.fit_regularized(method='l1',alpha=1/C1) 
print result.summary()

# %%

ix = [ix for ix,i in sorted(enumerate(classifier.predict_proba(X_test)[:,1]), key=lambda x:x[1], reverse=True) ]

sorted_ix = X_test.index[ix]
print sorted_ix
plt.figure()
plt.plot(preds[ix],[data_test[data_test.index==i]['days_in_company'].values[0] for i in sorted_ix],ls='',marker='.')
plt.ylabel('Days in company',fontsize=16)
plt.xlabel('Probability of churn at 3 months \n[based on initial 1 month features]',fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.tight_layout()
plt.savefig('predicted_test_prob.png')

# %%
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
#    plt.title(title)
#    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",fontsize=24)

    plt.ylabel('True label',fontsize=20)
    plt.xlabel('Predicted label',fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=20)

    plt.tight_layout()


class_names = ['Not churned','Churned']
cnf_matrix = confusion_matrix(y_test, y_pred)

np.set_printoptions(precision=2)


# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')
plt.tight_layout()
## Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')
plt.tight_layout()