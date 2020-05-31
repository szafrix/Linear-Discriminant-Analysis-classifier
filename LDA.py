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
import math

class LDA_single_feat():
    def fit(self, data, feature, attributes):
        self.feat = feature
        self.atts_name = attributes
        self.atts = sorted(data[self.atts_name].unique())
    # calculate prior probabilities
        self.priors = [data[data[self.atts_name]==i].shape[0]/data.shape[0] for i in self.atts]
    # calculate means and variances
        self.means = [data[data[self.atts_name]==i][self.feat].mean() for i in self.atts]
        self.var = [data[data[self.atts_name]==i][self.feat].var() for i in self.atts]
    def predict(self, data):
        LDA_scores = []
        for obs in data.index:
            feat = data[self.feat][obs]
            scores = [(feat*(self.means[i]/self.var[i])) - (self.means[i]**2)/(2*self.var[i]) + math.log(self.priors[i]) for i in range(len(self.atts))]
            LDA_scores.append(scores.index(max(scores)))
        return LDA_scores

class LDA_multivariate():
    def fit(self, data, features, attributes):
        self.feat_names = features
        self.atts_name = attributes
        self.atts = sorted(data[attributes].unique())
        self.priors = [data[data[self.atts_name]==i].shape[0]/data.shape[0] for i in self.atts]
        self.means = [data[data[self.atts_name]==i][self.feat_names].mean().values for i in self.atts]
        self.variances = [data[data[self.atts_name]==i][self.feat_names].var().values for i in self.atts]
        self.covariance_matrix = np.cov(data[self.feat_names], rowvar=False)
    
    def predict(self, data):
        LDA_scores = []
        for obs in data.index:
            feats = data[self.feat_names].loc[obs].values
            partone = [np.matmul(np.matmul(np.transpose(feats), np.linalg.inv(self.covariance_matrix)), self.means[i]) for i in range(len(self.atts))]
            parttwo = [np.matmul(np.matmul(0.5*np.transpose(self.means[i]), np.linalg.inv(self.covariance_matrix)), self.means[i]) for i in range(len(self.atts))]
            partthree = [math.log(i) for i in self.priors]
            scores = [partone[i] - parttwo[i] + partthree[i] for i in range(len(partone))]
            LDA_scores.append(scores.index(max(scores)))
        return LDA_scores