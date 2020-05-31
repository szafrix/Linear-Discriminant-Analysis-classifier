# -*- coding: utf-8 -*-
"""
Created on Sun May 31 14:32:22 2020

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
from sklearn.cluster import KMeans

np.random.seed(42)

"""
2 features
both gaussian distributed
lda assumes same distribution so we'll standardize them
"""

data = pd.DataFrame(np.random.multivariate_normal([10,20], [[20, 7], [7, 20]], size=500), columns=['feat1', 'feat2'])
data['atts'] = KMeans(n_clusters=3).fit(data).labels_

for i in data['atts'].unique():
    plt.scatter(data[data['atts']==i]['feat1'], data[data['atts']==i]['feat2'])
plt.show()

train_set, test_set = train_test_split(data)

LDA_model = LDA_multivariate()
LDA_model.fit(train_set, ['feat1', 'feat2'], 'atts')
train_set['LDA_class'] = LDA_model.predict(train_set)
test_set['LDA_class'] = LDA_model.predict(test_set)
print(f"Accuracy score of LDA on: train set: {accuracy_score(train_set['atts'], train_set['LDA_class'])}\n\
      test set: {accuracy_score(test_set['atts'], test_set['LDA_class'])}")

from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression()
lr_model.fit(train_set[['feat1', 'feat2']], train_set['atts'])
train_set['Logreg_score'] = lr_model.predict(train_set[['feat1', 'feat2']])
test_set['Logreg_score'] = lr_model.predict(test_set[['feat1', 'feat2']])
print(f"Accuracy score of Logistic Regression on: train set: {accuracy_score(train_set['atts'], train_set['Logreg_score'])}\n\
      test set: {accuracy_score(test_set['atts'], test_set['Logreg_score'])}")