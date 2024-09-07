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

from sklearn.metrics import accuracy_score

import tensorflow as tf

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
    df = pd.read_csv(os.path.join(config.train_dir, "W05_S006.csv"), header=0)

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

    model = Sequential()

    model.add(Dense(32, activation="relu", input_dim=30))
    model.add(Dense(12, activation="softmax", input_dim=32))

    optimizer = SGD(learning_rate=0.001)

    model.compile(loss='categorical_crossentropy', optimizer = optimizer)

    history = model.fit(X_train, Y_train, epochs=80, verbose=0, validation_data=(X_test, Y_test))

    loss_train = np.pad(history.history["loss"], (0, max(0, num_epochs - len(history.history["loss"]))), mode="edge")
    loss_valid = np.pad(history.history["val_loss"], (0, max(0, num_epochs - len(history.history["val_loss"]))), mode="edge")

    plt.plot(loss_train, label=f"Train Loss")
    plt.plot(loss_valid, label=f"Valid Loss")

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Convergence Plot per Training Cycle')
    plt.legend()
    plt.show()

    print(df)
    '''