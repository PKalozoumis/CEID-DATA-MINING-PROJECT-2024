import pandas as pd
import os
import numpy as np
import json
import sys
from collections import namedtuple
import time
import re

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import KBinsDiscretizer

import evaluate

#====================================================================================================

pd.options.mode.chained_assignment = None  # default='warn'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress only warnings (not errors or info)

config = None

with open("config.json", "r") as f:
    temp = json.load(f)
    Config = namedtuple("Config", temp.keys())
    config = Config(**temp)

#====================================================================================================

def neural_network(df, model_name):
    X = df.drop(columns=["label", "participant"])
    Y = df[["label"]]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1997)

    model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=200, activation="relu", random_state=1997)

    print("Training model...\n")
    t = time.time()
    model.fit(X_train,Y_train)
    print(f"\nTime: {time.time() - t}s\n")

    evaluate.dump_model(model_name, model, X_test, Y_test)

    predictions, matrix = evaluate.predict(model, X_test, Y_test)
    evaluate.evaluate(matrix, model_name)

#=============================================================================================================

def bayes(df, model_name):
    #Binning
    #------------------------------------------------------------------------------------------
    X = df.drop(columns=["label", "participant"])
    Y = df[["label"]]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1997)
    #data_train, data_test = train_test_split(df.drop(columns=["participant"]), test_size=0.3, random_state=1997)

    scaler = KBinsDiscretizer(n_bins=20, encode='ordinal', strategy='quantile')

    scaler.fit_transform(X_train)

    X_train = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    #Bayes
    #------------------------------------------------------------------------------------------

    model = CategoricalNB()

    print("Training model...\n")
    t = time.time()
    model.fit(X_train,Y_train)
    print(f"\nTime: {time.time() - t}s\n")
    
    evaluate.dump_model(model_name, model, X_test, Y_test)

    predictions, matrix = evaluate.predict(model, X_test, Y_test)
    evaluate.evaluate(matrix, model_name)

#====================================================================================================

if __name__ == "__main__":

    opt = None
    err = False

    while (opt == None):
        print("\nPlease choose the type of model\n1. Bayesian\n2. Neural Network\n3. Random Forest\n")

        opt = input(f"{'Your option' if not err else 'Invalid option, try again'}: ")

        if not re.match(r"1|2|3", opt):
            err=True
            opt = None
            print("\033[7A", end='')
            print("\033[J", end='')

    opt = int(opt)

    model_name = input("\nName of your model: ")
    print()

    #Initialize dataset
    #=======================================================================================

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
        #regex = r"W03_S006.csv"

        if not re.match(regex, file):
            continue

        print(f"Reading {file}...")

        df = pd.read_csv(os.path.join(config.train_dir, file), index_col=None, header=0, usecols = columns)
        
        df["participant"] = int(file[6:8])
        df = df.astype({"label": "int"})
        li.append(df)

    df = pd.concat(li, axis=0, ignore_index=True)

    print()

    if opt == 1:
        bayes(df, model_name)
    elif opt == 2:
        neural_network(df, model_name)
    else:
        pass

    
