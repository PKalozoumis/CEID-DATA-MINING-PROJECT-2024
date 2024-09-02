import pandas as pd
import os
import numpy as np
import shutil
import json
from collections import namedtuple
import re

#====================================================================================================

def file_to_window(file: str, window_size: int = 5) -> pd.DataFrame:

    df = pd.read_csv(file, index_col=None, header=0, usecols=["back_x", "back_y", "back_z", "thigh_x", "thigh_y", "thigh_z", "label"])

    s = df["label"].ne(df["label"].shift()).cumsum()
    dflist = [(df.iloc[0]["label"], df) for _,df in df.groupby(s)]

    features = []

    for label, df in dflist:

        if window_size > len(df):
            window_data = df.shift(periods=len(df) - window_size, fill_value=0.0)
            feature_vector = window_data.drop(columns="label").values.flatten()
            features.append(np.append(feature_vector, label))

        else:
            for i in range(window_size, len(df)+1):
                window_data = df.iloc[i-window_size:i]
                feature_vector = window_data.drop(columns="label").values.flatten()
                features.append(np.append(feature_vector, label))

    return features

    return pd.DataFrame(features, columns=columns)

#====================================================================================================

if __name__ == "__main__":

    config = None

    with open("config.json", "r") as f:
        temp = json.load(f)
        Config = namedtuple("Config", temp.keys())
        config = Config(**temp)

    window_size = 5
    
    #Feature names
    feature_names = ["back_x", "back_y", "back_z", "thigh_x", "thigh_y", "thigh_z"]

    columns = []

    #New column names
    for i in range(window_size):
        columns.append([n + f"{'_' + str(-(window_size-1 - i)) if (i < window_size-1) else ''}" for n in feature_names])

    columns = [elem for li in columns for elem in li]
    columns.append("label")

    #Make dataset

    os.makedirs(config.train_dir, exist_ok=True)

    for file in os.listdir(config.dataset_dir):

        fname = "W_" + file

        if os.path.exists(os.path.join(config.train_dir, fname)):
            print(f"Skipping file {file}...")
            continue

        print(f"Reading file {file}...")

        features = file_to_window(os.path.join(config.dataset_dir, file), window_size)
        df = pd.DataFrame(features, columns=columns)
        df = df.astype({"label": "int"},)
        df["participant"] = int(file[2:4])
        
        df.to_csv(os.path.join(config.train_dir, fname), index=False)