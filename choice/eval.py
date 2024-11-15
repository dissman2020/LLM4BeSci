#!/usr/bin/env python
# coding: utf-8
import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = 'FALSE'
# # Predicting choice behavior with extracted features
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split

import torch

# 从 .pth 文件读取保存的对象
choice = torch.load('choice.pth')
features = torch.load('features.pth')

# Initializing logistic regression
clf = LogisticRegressionCV(cv=10)

# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(features, choice, test_size=.1, random_state=42)
print(f'Train size: {len(X_train)}, test size: {len(X_test)}')

# Scaling the data
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Fitting the model and evaluating performance (this may take several minutes)
clf.fit(X_train, y_train)
torch.save(clf,"clf.pth")
print(f'Accuracy = {clf.score(X_test, y_test)}')
