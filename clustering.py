import json
from collections import namedtuple
import os
import sys

import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
from sklearn.metrics import silhouette_score

if __name__ == "__main__":
    config = None

    with open("config.json", "r") as f:
        temp = json.load(f)
        Config = namedtuple("Config", temp.keys())
        config = Config(**temp)

    df = None

    if not os.path.exists("proj/clustering.csv"):

        data = []

        all_activities = [1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 130, 140]
        features = ["back_x", "back_y", "back_z", "thigh_x", "thigh_y", "thigh_z"]

        for file in os.listdir(config.dataset_dir):
            print(f"Reading {file}...")

            df = pd.read_csv(os.path.join(config.dataset_dir, file), index_col=None, header=0, usecols=["back_x", "back_y", "back_z", "thigh_x", "thigh_y", "thigh_z", "label"])

            #Summarize the entire file in 72 dimensions
            record = {}
            performed_activities = df["label"].unique().tolist()

            for activity in all_activities:
                if activity in performed_activities:

                    temp = df.loc[df["label"] == activity]

                    for feature in features:
                        record[feature + "_" + str(activity)] = temp[feature].median()

                else:
                    for feature in features:
                        record[feature + "_" + str(activity)] = 0

            record["participant"] = int(file[2:4])

            data.append(record)

        df = pd.DataFrame(data)
        df.set_index("participant", inplace=True)
        df.to_csv("proj/clustering.csv", index=True)

    else:
        df = pd.read_csv("proj/clustering.csv", index_col="participant")

    #==============================================================================================================================

    clustering = AgglomerativeClustering(n_clusters=3)
    model = clustering.fit(df)

    print(model.labels_)

    #linked = linkage(df, method='ward')

    score = silhouette_score(df, model.labels_)
    print(f"Silhouette Score: {score}")

    sys.exit()

    # Plot the dendrogram
    plt.figure(figsize=(10, 7))
    dendrogram(model,
            orientation='top',
            labels=range(1, 73),
            distance_sort='descending',
            show_leaf_counts=True)
    


    print(labels)