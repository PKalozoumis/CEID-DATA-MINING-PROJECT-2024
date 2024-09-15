from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.metrics import silhouette_score
import hdbscan

import json
from collections import namedtuple
import os

#===============================================================================================================

if __name__ == "__main__":
    config = None

    with open("config.json", "r") as f:
        temp = json.load(f)
        Config = namedtuple("Config", temp.keys())
        config = Config(**temp)

    df = None

    os.makedirs(config.clustering_dir, exist_ok=True)

    #Read the clustering file if it exists
    #Otherwise, make it
    #===============================================================================================================
    if not os.path.exists(os.path.join(config.clustering_dir, "clustering.csv")):

        data = []

        all_activities = [1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 130, 140]
        features = ["back_x", "back_y", "back_z", "thigh_x", "thigh_y", "thigh_z"]

        for file in os.listdir(config.dataset_dir):
            print(f"Reading {file}...")

            df = pd.read_csv(os.path.join(config.dataset_dir, file), index_col=None, header=0, usecols=["back_x", "back_y", "back_z", "thigh_x", "thigh_y", "thigh_z", "label"])

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
        df.to_csv(os.path.join(config.clustering_dir, "clustering.csv"), index=True)
        print()

    else:
        df = pd.read_csv(os.path.join(config.clustering_dir, "clustering.csv"), index_col="participant")

    participants = df.index.tolist()

    #Hierarchical Clustering
    #===============================================================================================================
    print("Hierarchical Clustering\n============================================================================")

    num_clusters = 2

    # Generate the linkage matrix
    Z = linkage(df, method='ward')
    labels = fcluster(Z, t=num_clusters, criterion='maxclust')

    #Organize each cluster elements into lists
    clusters = [[] for _ in range(num_clusters)]

    for participant, cluster in zip(participants, labels):
        clusters[cluster-1].append(participant)

    for i, cluster in enumerate(clusters):
        print(f"Cluster {i:02}: {cluster}")

    #Evaluate clustering
    score = silhouette_score(df, labels)
    print(f"\nSilhouette Score: {score:.3f}")

    # Plot the dendrogram
    plt.figure(figsize=(10, 7))
    dendrogram(Z,
            orientation='top',
            labels=participants,
            distance_sort='descending',
            show_leaf_counts=True)
    
    plt.savefig(os.path.join(config.clustering_dir, "dendogram.png"))
    plt.close()

    #HDBSCAN
    #===============================================================================================================
    print("\nHDBSCAN Clustering\n============================================================================")

    clustering = hdbscan.HDBSCAN(min_cluster_size=2)

    model = clustering.fit(df)

    #Organize each cluster elements into lists
    num_clusters = len(np.unique(model.labels_[model.labels_ != -1]))

    clusters = [[] for _ in range(num_clusters)]

    for participant, cluster in zip(participants, model.labels_):
        clusters[cluster].append(participant)

    for i, cluster in enumerate(clusters):
        print(f"Cluster {i:02}: {cluster}")

    #Evaluate clustering
    score = silhouette_score(df, model.labels_)
    print(f"\nSilhouette Score: {score:.3f}")