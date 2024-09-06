import pandas as pd
import os
import numpy as np
import shutil
import json
from collections import namedtuple
import re
import argparse

#====================================================================================================

def file_to_window(file: str, window_size: int = 5) -> pd.DataFrame:

    attrs = ["back_x", "back_y", "back_z", "thigh_x", "thigh_y", "thigh_z", "label"]
    df = pd.read_csv(file, index_col=None, header=0, usecols=["back_x", "back_y", "back_z", "thigh_x", "thigh_y", "thigh_z", "label"])

    #This little trick or whatever will group SEQUENTIAL records that have the same label
    s = df["label"].ne(df["label"].shift()).cumsum()
    dflist = [(df.iloc[0]["label"], df) for _,df in df.groupby(s)]

    features = []

    for label, df in dflist:

        if window_size > len(df):
            shift = pd.DataFrame([[0]*len(attrs)], columns=attrs)
            shift["label"] = label
            new_rows = [shift]*(window_size - len(df))
            
            window_data = pd.concat([*new_rows, df])
            
            feature_vector = window_data.drop(columns="label").values.flatten()
            features.append(np.append(feature_vector, label))

        else:
            for i in range(window_size, len(df)+1):
                window_data = df.iloc[i-window_size:i]
                feature_vector = window_data.drop(columns="label").values.flatten()
                features.append(np.append(feature_vector, label))

    return features

#====================================================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Making windowed data out of the original dataset', allow_abbrev=False)
    parser.add_argument('--all', action="store_true", default=False, help="Recreate all files")
    args = parser.parse_args()

    config = None

    with open("config.json", "r") as f:
        temp = json.load(f)
        Config = namedtuple("Config", temp.keys())
        config = Config(**temp)

    window_size = 5
    
    #Feature names
    #==================================================================================================
    feature_names = ["back_x", "back_y", "back_z", "thigh_x", "thigh_y", "thigh_z"]

    columns = []

    #New column names
    #==================================================================================================
    for i in range(window_size):
        columns.append([n + f"{'_m' + str(window_size-1 - i) if (i < window_size-1) else ''}" for n in feature_names])

    columns = [elem for li in columns for elem in li]
    columns.append("label")

    #Make dataset
    #==================================================================================================
    os.makedirs(config.train_dir, exist_ok=True)

    for file in os.listdir(config.dataset_dir):

        fname = f"W{window_size:02}_{file}"

        if not args.all and os.path.exists(os.path.join(config.train_dir, fname)):
            print(f"Skipping file {file}...")
            continue

        print(f"Reading file {file}...")

        features = file_to_window(os.path.join(config.dataset_dir, file), window_size)
        df = pd.DataFrame(features, columns=columns)

        df = df.astype({"label": "int"},)
        df["participant"] = int(file[2:4])
        
        df.to_csv(os.path.join(config.train_dir, fname), index=False)