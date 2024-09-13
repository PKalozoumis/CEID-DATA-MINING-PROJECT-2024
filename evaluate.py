import argparse
from joblib import load, dump
import json
from collections import namedtuple
import os
import pandas as pd
import copy

from sklearn.metrics import accuracy_score, classification_report

#====================================================================================================

config = None

with open("config.json", "r") as f:
    temp = json.load(f)
    Config = namedtuple("Config", temp.keys())
    config = Config(**temp)

#====================================================================================================

def dump_model(name: str, model, X_test: pd.DataFrame, Y_test: pd.DataFrame):
    print("Dumping model...")
    dump(model, os.path.join(config.models_dir, f"{name}.joblib"))

    temp = copy.deepcopy(X_test)
    temp["label"] = Y_test["label"]
    temp.to_csv(os.path.join(config.models_dir, f"{name}.test"), index=False)

#====================================================================================================

def load_model(name: str):

    data = pd.read_csv(os.path.join(config.models_dir, f"{name}.test"))
    Y_test = data["label"]
    X_test = data.drop(columns=["label"])

    return load(os.path.join(config.models_dir, f"{name}.joblib")), X_test, Y_test

#====================================================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Evaluation of classification model', allow_abbrev=False)
    parser.add_argument('model', action="store", default=None, help="Model name (e.g. \"bayes\", without the .joblib extension). Must be under models directory")
    args = parser.parse_args()

    model, X_test, Y_test = load_model(args.model)
    print(model)

    #Evaluate
    #==========================================================================================
    predictions = model.predict(X_test)
    print(classification_report(Y_test, predictions))

