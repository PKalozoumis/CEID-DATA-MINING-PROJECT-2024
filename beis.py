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

pd.options.mode.chained_assignment = None  # default='warn'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress only warnings (not errors or info)

#====================================================================================================

if __name__ == "__main__":

    config = None
    num_epochs = 80

    with open("config.json", "r") as f:
        temp = json.load(f)
        Config = namedtuple("Config", temp.keys())
        config = Config(**temp)

    #Initialize dataset
    #=======================================================================================
    #df = pd.read_csv(os.path.join(config.train_dir, "W05_S006.csv"), header=0)

    li = []

    for file in os.listdir(config.train_dir):
        print(f"Reading {file}...")

        df = pd.read_csv(os.path.join(config.train_dir, file), index_col=None, header=0)
        
        match = re.match(r"W05_S0([\d]{2})\.csv", file)
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

    model.fit(X_train,Y_train)
    predictions = model.predict(X_test)

    #true_labels = data_test["label"].tolist()

    #accuracy = accuracy_score(true_labels, predictions)
    #print(f"Accuracy: {accuracy}")
    print(classification_report(Y_test, predictions))

    '''
    t = time.time()

    hc = HillClimbSearch(binned)
    best_model = hc.estimate(scoring_method=BicScore(binned))

    print(best_model)

    # Define Bayesian Network model
    model = BayesianNetwork(best_model.edges())

    model.fit(binned, estimator=MaximumLikelihoodEstimator)

    print("Nodes in the model:", model.nodes())

    inference = VariableElimination(model)

    test_binned = pd.DataFrame(scaler.transform(data_test.drop(columns=["label"])), columns=data_test.drop(columns=["label"]).columns)
    test_binned["label"] = data_test["label"]

    predictions = []
    for index, row in test_binned.iterrows():
        evidence = row.drop("label").to_dict()
        result = inference.map_query(variables=['label'], evidence=evidence)
        predictions.append(result)

    predictions = [int(val) for pred in predictions for val in pred.values() ]
    true_labels = data_test["label"].tolist()

    accuracy = accuracy_score(true_labels, predictions)
    print(f"Accuracy: {accuracy}")
    print(f"Time: {time.time() - t}s")
    '''