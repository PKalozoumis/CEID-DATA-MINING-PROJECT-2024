import argparse
from joblib import load, dump
import json
from collections import namedtuple
import os
import pandas as pd
import copy
import sys
import time

#====================================================================================================

config = None

with open("config.json", "r") as f:
    temp = json.load(f)
    Config = namedtuple("Config", temp.keys())
    config = Config(**temp)

#====================================================================================================

def dump_model(name: str, model, X_test: pd.DataFrame, Y_test: pd.DataFrame):
    os.makedirs(config.models_dir, exist_ok=True)

    print(f"Dumping model {name}.joblib...")
    dump(model, os.path.join(config.models_dir, f"{name}.joblib"))

    X_test = X_test.reset_index(drop=True)
    Y_test = Y_test.reset_index(drop=True)

    print(f"Dumping test data {name}.test...")
    temp = copy.deepcopy(X_test)
    temp["label"] = Y_test["label"].astype(int)
    temp.to_csv(os.path.join(config.models_dir, f"{name}.test"), index=False)

#====================================================================================================

def load_model(name: str):

    print(f"Loading model {name}.joblib...")
    model = load(os.path.join(config.models_dir, f"{name}.joblib"))

    print(f"Loading test data {name}.test...")
    data = pd.read_csv(os.path.join(config.models_dir, f"{name}.test"))

    Y_test = data[["label"]]
    X_test = data.drop(columns=["label"])

    return model, X_test, Y_test

#====================================================================================================

def predict(model, X_test, Y_test):

     #Construct index
     #------------------------------------------------------------------------------------------
    index_arr = []

    labels = sorted(Y_test["label"].unique())

    for label in labels:
        index_arr.append((label, "p"))
        index_arr.append((label, "n"))

    index = pd.MultiIndex.from_tuples(index_arr, names=["label", "prediction"])
    confusion = pd.DataFrame(columns=["p", "n"], index=index)

    #Predict
    #------------------------------------------------------------------------------------------
    print("Predicting...")
    predictions = model.predict(X_test)

    #Make confusion matrix
    #------------------------------------------------------------------------------------------
    
    print("Calculating confusion matrices...")

    compare = pd.DataFrame({"pred": predictions, "true": Y_test["label"].tolist()})

    for label in labels:
        temp = compare.loc[compare["pred"]==label]["true"]
        tp = temp[temp == label].count()
        fp = temp[temp != label].count()

        temp = compare.loc[compare["pred"]!=label]["true"]
        fn = temp[temp == label].count()
        tn = temp[temp != label].count()

        confusion.loc[(label, "p")] = [tp, fp]
        confusion.loc[(label, "n")] = [fn, tn]

    return predictions, confusion

#====================================================================================================

def evaluate(matrix, model_name):

    os.makedirs(config.evaluation_dir, exist_ok=True)

    scores = pd.DataFrame(index=matrix.index.get_level_values("label").unique(), columns=["precision", "recall", "fscore"])

    all = 0
    correct = 0

    for label, data in matrix.groupby(level="label"):

        data = data.droplevel("label")

        tp = data.loc["p", "p"]
        fp = data.loc["p", "n"]
        fn = data.loc["n", "p"]
        tn = data.loc["n", "n"]

        scores.loc[label, "precision"] =  round((tp)/(tp+fp), 2)
        scores.loc[label, "recall"] = round((tp)/(tp+fn), 2)
        scores.loc[label, "fscore"] = round((2*tp)/(2*tp+fp+fn), 2)
        
        all = tp+tn+fp+fn
        correct += tp

    accuracy = round(correct/all, 2)

    #Write results to a file
    if model_name is not None:

        fname = os.path.join(config.evaluation_dir, f"{model_name}.xlsx")

        while(True):
            try:
                with pd.ExcelWriter(fname, engine="xlsxwriter") as writer:
                    scores.to_excel(writer)

                    worksheet = writer.sheets["Sheet1"]
                    row_start = len(scores) + 2
                    worksheet.write(row_start, 0, "Accuracy")
                    worksheet.write(row_start, 1, accuracy)

                break
            except PermissionError:
                print(f"Please close the file {fname}")
                time.sleep(1)

        print(f"Saved evaluation metrics at {fname}")

    return scores, accuracy

#====================================================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Evaluation of classification model', allow_abbrev=False)
    parser.add_argument('model', action="store", default=None, help="Model name (e.g. \"bayes\", without the .joblib extension). Must be under models directory")
    args = parser.parse_args()

    model, x, y = load_model(args.model)
    predictions, confusion = predict(model, x, y)

    print()
    metrics, accuracy = evaluate(confusion, args.model)
    print()
    print(metrics)
    print(f"\nAccuracy: {accuracy}")