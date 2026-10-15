# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 16:23:31 2022

@author: user

Shared settings and data preparation. Every other module does
``from EDA import *``, so the names defined here are the project's globals.
"""

import os
import random
import time
from copy import deepcopy
from operator import attrgetter
from pathlib import Path
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, InputLayer
from keras.models import Sequential
from sklearn import metrics
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils import class_weight

# Defining the seed, and applying it everywhere
SEED = 2022
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Define Hyperparameters for NN
HIDDEN_LAYER_COUNT = [1, 2, 3, 4, 5]
HIDDEN_LAYER_NEURONS = [8, 16, 24, 32, 64, 128, 256, 512]
HIDDEN_LAYER_RATE = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
HIDDEN_LAYER_ACTIVATIONS = ['tanh', 'relu', "sigmoid", "softplus"]
HIDDEN_LAYER_TYPE = ['dense', 'dropout']
MODEL_OPTIMIZER = ['adam', "rmsprop"]
MODEL_LOSS = [tf.keras.losses.BinaryCrossentropy(),
              tf.keras.losses.BinaryFocalCrossentropy(gamma=3.0)]


# Define Genetic Algorithm Parameters. The environment variables let CI run a
# small configuration; the defaults are the 2022 settings.
GENERATIONS = int(os.environ.get("GA_GENERATIONS", 20))
SIZE = int(os.environ.get("GA_SIZE", 12))
ELITE_SIZE = 2
TOURNAMENT_SIZE = 3
CROSSOVER_PROBABILITY = 0.7
MUTATION_PROBABILITY = 0.2
OPTIM = "max"
PATIENCE = 5
EPOCHS = int(os.environ.get("GA_EPOCHS", 100))
THRESHOLD = 0.5

DATA_URL = ("https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
            "master/data/Telco-Customer-Churn.csv")
DATA_PATH = Path(__file__).resolve().parent / "data" / "Telco-Customer-Churn.csv"


def load_data():
    if not DATA_PATH.exists():
        DATA_PATH.parent.mkdir(exist_ok=True)
        urlretrieve(DATA_URL, DATA_PATH)
    return pd.read_csv(DATA_PATH)


def remove_correlated(X, threshold=0.9):
    corr = np.absolute(X.corr(numeric_only=True))
    # Select upper triangle of correlation matrix
    corr_matrix = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    # Find features with correlation greater than threshold
    correlated_cols = [c for c in corr_matrix.columns if any(corr_matrix[c] > threshold)]
    for col in correlated_cols:
        print(f"{col} is highly correlated with an earlier column")
    return correlated_cols


def remove_outliers(x_train, y_train, cols, estimators=100):
    '''
    Uses an isolation forest to detect and remove outliers from the
    training set only. Validation and test rows are never dropped.
    '''
    isolation_forest = IsolationForest(n_estimators=estimators, random_state=SEED)
    inliers = isolation_forest.fit_predict(x_train[cols]) == 1
    return x_train[inliers], y_train[inliers]


def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)


ds = load_data()

ds['TotalCharges'] = pd.to_numeric(ds['TotalCharges'], errors='coerce')
ds.dropna(inplace=True)

# Droping the column 'CustomerID' since it is not useful:
ds.drop(columns=['customerID'], inplace=True)

ds["Churn"] = np.where(ds["Churn"] == 'Yes', 1, 0)
y = ds["Churn"]
X = ds.drop(columns=["Churn"])

# Split into train (60%), validation (20%) and test (20%). The GA scores
# architectures on validation; test is touched once, for the final model.
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, stratify=y, random_state=SEED)
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.25, shuffle=True, stratify=y_train, random_state=SEED)

correlated_cols = remove_correlated(x_train)
x_train = x_train.drop(columns=correlated_cols)
x_val = x_val.drop(columns=correlated_cols)
x_test = x_test.drop(columns=correlated_cols)

num_features = [c for c in ["tenure", "MonthlyCharges", "TotalCharges"] if c in x_train.columns]
x_train, y_train = remove_outliers(x_train, y_train, num_features)

print("train / validation / test:", x_train.shape, x_val.shape, x_test.shape)

scaler = StandardScaler()
x_train[num_features] = scaler.fit_transform(x_train[num_features])
x_val[num_features] = scaler.transform(x_val[num_features])
x_test[num_features] = scaler.transform(x_test[num_features])

cat_features = sorted(set(x_train.columns).difference(num_features).difference(["SeniorCitizen"]))

# One-hot encoding for categorical features:
encoder = OneHotEncoder(drop="if_binary", sparse_output=False, handle_unknown="ignore")
x_train_cat = encoder.fit_transform(x_train[cat_features])
x_val_cat = encoder.transform(x_val[cat_features])
x_test_cat = encoder.transform(x_test[cat_features])

x_train = np.concatenate((x_train_cat, x_train[num_features].values), axis=1)
x_val = np.concatenate((x_val_cat, x_val[num_features].values), axis=1)
x_test = np.concatenate((x_test_cat, x_test[num_features].values), axis=1)
y_train, y_val, y_test = y_train.values, y_val.values, y_test.values
