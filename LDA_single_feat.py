# -*- coding: utf-8 -*-
"""
Created on Sat May 30 20:44:14 2020

@author: Szafran
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.metrics import accuracy_score
import seaborn as sns
from LDA import *

pd.set_option('chained_assignment', None)

np.random.seed(42)
positives = pd.DataFrame(np.transpose([np.random.normal(10, 3, 1000), np.ones(1000)]), columns=['feats', 'atts'])
negatives = pd.DataFrame(np.transpose([np.random.normal(5, 3, 1000), np.zeros(1000)]), columns=['feats', 'atts'])


data = pd.concat([positives, negatives] , axis=0, ignore_index=True)
train_set, test_set = train_test_split(data)
# histograms of features 
plt.hist(train_set[train_set['atts']==1]['feats'], bins=20, histtype='bar', color='g', label='atts=1')
plt.hist(train_set[train_set['atts']==0]['feats'], bins=20, histtype='step', color='r', label='atts=0')
plt.legend()
plt.title('Histograms of feats')
plt.show()
# check normality of distributions, kolmogorow-smirnow test
def check_gaussian(data):
    return stats.kstest(data.values, stats.norm(data.mean(), data.std()).cdf).pvalue

print(f"P-value of KS test for the distributions of features given the attribute value:\n\
      {check_gaussian(data[data['atts']==1]['feats'])},\
      {check_gaussian(data[data['atts']==0]['feats'])}")


#model creation and fit
LDA_model = LDA_single_feat()
LDA_model.fit(train_set, 'feats', 'atts')
# predictions on both train and test sets
train_set['LDA_scores'] = LDA_model.predict(train_set)
test_set['LDA_scores'] = LDA_model.predict(test_set)

print(f"Accuracy score of LDA, train set: {accuracy_score(train_set['atts'], train_set['LDA_scores'])}.")
print(f"Accuracy score of LDA, test set: {accuracy_score(test_set['atts'], test_set['LDA_scores'])}.")


# porównanie z regresją logistyczną
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression()
lr_model.fit(train_set['feats'].values.reshape(-1,1), train_set['atts'])
train_set['logreg'] = lr_model.predict(train_set['feats'].values.reshape(-1,1))
test_set['logreg'] = lr_model.predict(test_set['feats'].values.reshape(-1,1))

print(f"Accuracy score of logreg, train set: {accuracy_score(train_set['atts'], train_set['logreg'])}.")
print(f"Accuracy score of logreg, test set: {accuracy_score(test_set['atts'], test_set['logreg'])}.")

#LDA slightly outperforms logistic regression on the train set
plt.figure(figsize=(10,7))
plt.title("Distribution of features grouped by attributes' values")
plt.xlabel('Feature value')
plt.ylim(0,0.135)
sns.kdeplot(data[data['atts']==1]['feats'], label="atts=1", color='g')
sns.kdeplot(data[data['atts']==0]['feats'], label='atts=0', color='r')
plt.vlines(train_set[train_set['LDA_scores']==0]['feats'].max(), 0, 0.14, color='y',
           label='boundary for LDA', linestyles='dashed')
plt.vlines(train_set[train_set['logreg']==0]['feats'].max(), 0, 0.14, color='b',
           label='boundary for Logistic Regression', linestyles='dashed')
plt.legend(loc=1)
plt.show()