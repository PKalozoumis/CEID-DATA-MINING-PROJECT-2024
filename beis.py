import pandas as pd
import os
import numpy as np
import json
import sys
from collections import namedtuple

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import KBinsDiscretizer

import time

from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BicScore, HillClimbSearch, MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

from sklearn.metrics import accuracy_score, classification_report

import tensorflow as tf

import re
import evaluate

from joblib import load, dump

pd.options.mode.chained_assignment = None  # default='warn'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress only warnings (not errors or info)

config = None

with open("config.json", "r") as f:
    temp = json.load(f)
    Config = namedtuple("Config", temp.keys())
    config = Config(**temp)

#====================================================================================================

if __name__ == "__main__":

    num_epochs = 80

    #Initialize dataset
    #=======================================================================================
    #df = pd.read_csv(os.path.join(config.train_dir, "W05_S006.csv"), header=0)

    #Limit dataset
    #It's possible files with bigger window sizes were generated
    #Because big window sizes lead to very slow training, you have the option to only use part of the data
    #-------------------------------------------------------------------------------------

    window_size = 3

    feature_names = ["back_x", "back_y", "back_z", "thigh_x", "thigh_y", "thigh_z"]

    columns = []
    for i in range(window_size):
        columns.append([n + f"{'_m' + str(window_size-1 - i) if (i < window_size-1) else ''}" for n in feature_names])

    columns = [elem for li in columns for elem in li]
    columns.append("label")
    #-------------------------------------------------------------------------------------

    li = []

    for file in os.listdir(config.train_dir):

        regex = r"W" + re.escape(f"{window_size:02}") + r"_S0[\d]{2}\.csv"

        if not re.match(regex, file):
            continue

        print(f"Reading {file}...")

        df = pd.read_csv(os.path.join(config.train_dir, file), index_col=None, header=0, usecols = columns)
        
        df["participant"] = int(file[6:8])
        df = df.astype({"label": "int"})
        li.append(df)

    df = pd.concat(li, axis=0, ignore_index=True)

    X = df.drop(columns=["label", "participant"])
    Y = df[["label"]]
    
    labels = [1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 130, 140]

    for label in labels:
        Y["out" + str(label)] = (Y["label"] == label).apply(int)

    Y.drop("label", inplace=True, axis=1)

    #Binning
    #------------------------------------------------------------------------------------------

    data_train, data_test = train_test_split(df.drop(columns=["participant"]), test_size=0.3, random_state=1997)

    scaler = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')

    scaler.fit(data_train.drop(columns=["label"]))


    #print(pd.DataFrame(scaler.transform(X)))
    binned = pd.DataFrame(scaler.transform(data_train.drop(columns=["label"])), columns=data_train.drop(columns=["label"]).columns)
    binned["label"] = data_train["label"]

    #Bayesian network
    #=======================================================================================

    Y = df[["label"]]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1997)

    model = GaussianNB()

    t = time.time()

    model.fit(X_train,Y_train)
    
    evaluate.dump_model("bayes", model, X_test, Y_test)

    predictions = model.predict(X_test)

    #true_labels = data_test["label"].tolist()

    #accuracy = accuracy_score(true_labels, predictions)
    #print(f"Accuracy: {accuracy}")
    print(classification_report(Y_test, predictions))
    print(f"Time: {time.time() - t}s")