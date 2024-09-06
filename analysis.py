import pandas as pd
from matplotlib import pyplot as plt
import os
import seaborn as sns
import numpy as np
import re
import copy
import json
from collections import namedtuple
import sys
import math
from datetime import datetime
import argparse

import shutil

activities = {
    1: "walking",
    2: "running",
    3: "shuffling",
    4: "stairs ascending",
    5: "stairs descending",
    6: "standing",
    7: "sitting",
    8: "lying",
    13: "cycling sit",
    14: "cycling stand",
    130: "cycling sit inactive",
    140: "cycling stand inactive"
}

attr_list = ["back_x", "back_y", "back_z", "thigh_x", "thigh_y", "thigh_z"]
activity_list = [1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 130, 140]

# noinspection SpellCheckingInspection
if __name__ == "__main__":

    #Config and directory initialization
    #===================================================================================================================
    parser = argparse.ArgumentParser(description='Dataset Analysis', allow_abbrev=False)
    parser.add_argument('--tseries', action="store_true", default=False)
    parser.add_argument('--density', action="store_true", default=False)
    parser.add_argument('--corr', action="store_true", default=False)
    args = parser.parse_args()
    
    #Config and directory initialization
    #===================================================================================================================
    config = None

    with open("config.json", "r") as f:
        temp = json.load(f)
        Config = namedtuple("Config", temp.keys())
        config = Config(**temp)

    #Results directory
    #---------------------------------------------------------------------------------
    os.makedirs(config.results_dir, exist_ok=True)

    if os.path.exists(f"{config.results_dir}/other"):
        shutil.rmtree(f"{config.results_dir}/other")

    os.makedirs(f"{config.results_dir}/other")

    #Density
    #---------------------------------------------------------------------------------
    if args.density:
        if os.path.exists(f"{config.results_dir}/density"):
            shutil.rmtree(f"{config.results_dir}/density")

        if os.path.exists(f"{config.results_dir}/density/grid"):
            shutil.rmtree(f"{config.results_dir}/density/grid")

        os.makedirs(f"{config.results_dir}/density")
        os.makedirs(f"{config.results_dir}/density/grid")

    #Timeseries
    #---------------------------------------------------------------------------------
    if args.tseries:
        if os.path.exists(f"{config.results_dir}/timeseries"):
            shutil.rmtree(f"{config.results_dir}/timeseries")

        os.makedirs(f"{config.results_dir}/timeseries")

    #Correlation
    #---------------------------------------------------------------------------------

    os.makedirs(f"{config.results_dir}/correlation", exist_ok=True)

    if args.corr:

        if os.path.exists(f"{config.results_dir}/correlation"):
            shutil.rmtree(f"{config.results_dir}/correlation")

        if os.path.exists(f"{config.results_dir}/correlation/heatmaps"):
            shutil.rmtree(f"{config.results_dir}/correlation/heatmaps")

        if os.path.exists(f"{config.results_dir}/correlation/scatter"):
            shutil.rmtree(f"{config.results_dir}/correlation/scatter")

        os.makedirs(f"{config.results_dir}/correlation")
        os.makedirs(f"{config.results_dir}/correlation/heatmaps")
        os.makedirs(f"{config.results_dir}/correlation/scatter")    

    #Read files into one dataframe
    #===================================================================================================================
    li = []

    #for file in ["S006.csv"]:
    for file in os.listdir(config.dataset_dir):
        print(f"Reading {file}...")

        df = pd.read_csv(os.path.join(config.dataset_dir, file), index_col=None, header=0, usecols=["timestamp", "back_x", "back_y", "back_z", "thigh_x", "thigh_y", "thigh_z", "label"])
        
        match = re.match(r"S0([\d]{2})\.csv", file)
        df["participant"] = int(file[2:4])
        df = df.astype({"label": "int"})
        li.append(df)

    df = pd.concat(li, axis=0, ignore_index=True)

    #Stuff
    #===================================================================================================================

    print()

    labels = np.sort(df["label"].unique())

    data = df[["back_x", "back_y", "back_z", "thigh_x", "thigh_y", "thigh_z"]].astype("float64")
    data["label"] = df["label"]

    mean = data.groupby("label").mean()
    mean.to_excel(os.path.join(config.results_dir, "other", "mean.xlsx"), index_label="Label")
    print("Saving mean.xlsx...")

    median = data.groupby("label").median()
    median.to_excel(os.path.join(config.results_dir, "other", "median.xlsx"), index_label="Label")
    print("Saving median.xlsx...\n")

    corr = data.groupby("label").corr()
    corr.index.set_names(["label", "dim"], inplace=True)

    for label in labels:
        corr.loc[label].to_excel(os.path.join(config.results_dir, "correlation", f"corr_{label:03}.xlsx"))
        print(f"Saving corr_{label:03}.xlsx...\n")

    std = data.groupby("label").std()
    mean.to_excel(os.path.join(config.results_dir, "other", "std.xlsx"), index_label="Label")
    print("Saving std.xlsx...\n")

    #Time Series Plots
    #===================================================================================================================

    if args.tseries:
        #We want to have 5 time series examples for each activity
        tseries_counts = {key: 0 for key in activities.keys()}

        for participant in df["participant"].unique():

            done = True

            for val in tseries_counts.values():
                if val < 5:
                    done = False
                    break

            if done:
                break

            print(f"Participant {participant}")

            pdata = df.loc[df["participant"] == participant]

            #Split participant data based on consecutive same value on the label column
            #This will show us how many times the participant changed activity

            s = pdata["label"].ne(pdata["label"].shift()).cumsum()
            groups = [(pdata.iloc[0]["label"], pdata) for _, pdata in pdata.groupby(s)]

            for j, (label, group) in enumerate(groups):

                if tseries_counts[label] == 5:
                    continue

                print(f"Generating time series {tseries_counts[label]+1}/5 for {activities[label]}...")

                tfig, tax = plt.subplots(nrows=2, ncols=3, figsize=(16,9))

                for i, attr in enumerate(["back_x", "back_y", "back_z", "thigh_x", "thigh_y", "thigh_z"]):
                    tax[i//3, i%3].set_xticks([])
                    tax[i//3, i%3].plot(group["timestamp"], group[attr])
                    tax[i//3, i%3].set_xlabel("Timestamp")
                    tax[i//3, i%3].set_ylabel("Value")
                    tax[i//3, i%3].set_title(f'{attr}')

                start_time = datetime.strptime(group['timestamp'].iloc[0], "%Y-%m-%d %H:%M:%S.%f").time()
                end_time = datetime.strptime(group['timestamp'].iloc[-1], "%Y-%m-%d %H:%M:%S.%f").time()
                tfig.suptitle(f"Time series data for participant {participant} activity {label} ({activities[label]})\nStart: {start_time} End: {end_time}")

                tfig.savefig(os.path.join(config.results_dir, "timeseries", f"tseries_{activities[label].replace(' ', '_')}_{tseries_counts[label]}.png"))

                plt.close(tfig)

                #Update count
                tseries_counts[label] += 1

    #Plots
    #===================================================================================================================

    for label in np.sort(df["label"].unique()):

        current_label_data = data.loc[data["label"] == label, ["back_x", "back_y", "back_z", "thigh_x", "thigh_y", "thigh_z"]]

        if args.corr:

            print(f"Generating correlation heatmap for Activity {label:03}...")

            corr_table = corr.loc[label]

            #Correlation heatmap
            #-----------------------------------------------------------------------------------------------------------------------
            fig, ax = plt.subplots()
            sns.heatmap(corr_table, annot=True, cmap='coolwarm', fmt='.3f', ax = ax, vmin=-1, vmax=1)
            ax.set_title(f"Correlation heatmap for Activity {label} ({activities[label]})")
            fig.savefig(os.path.join(config.results_dir, "correlation", "heatmaps", f"corr_{label:03}.png"))

            plt.close(fig)

            #Scatter plots
            #-----------------------------------------------------------------------------------------------------------------------
            for dim, corr_data in corr_table.iterrows():
                stop = False
                for attr in attr_list:
                    if dim == attr:
                        break

                    if abs(corr_data[attr]) > 0.4:
                        plt.scatter(current_label_data[dim], current_label_data[attr])
                        plt.xlabel(dim)
                        plt.ylabel(attr)
                        plt.title(f"Scatter plot between {dim} and {attr} for Activity {label}\n(Correlation = {corr_data[attr]:.2f})")
                        plt.savefig(os.path.join(config.results_dir, "correlation", "scatter", f"scatter_{label:03}_{dim}_{attr}.png"), bbox_inches="tight")
                        plt.close()

        if args.density:

            print(f"Generating density plots for Activity {label:03}...")

            #Density plots
            #-----------------------------------------------------------------------------------------------------------------------

            fig0, ax0 = plt.subplots()
            fig_grid, ax_grid = plt.subplots(nrows = 2, ncols = 3, figsize=(16,9))

            for i, (name, col) in enumerate(current_label_data.items()):
                sns.kdeplot(col, fill=True, label=name, ax = ax0)
                sns.kdeplot(col ,fill=True, label=name, ax = ax_grid[i//3, i%3])

                ax_grid[i//3, i%3].set_title(f"Density Plot of {name} for Activity {label}\n({activities[label]})")
                ax_grid[i//3, i%3].set_xlabel("Value")
                ax_grid[i//3, i%3].set_ylabel("Density")

                #low = median.loc[label, name] - 3*std.loc[label, name]
                #high = median.loc[label, name] + 3*std.loc[label, name]

                #ax_grid[i//3, i%3].axvline(x=low, color='r', linestyle='-', linewidth=1)
                #ax_grid[i//3, i%3].axvline(x=high, color='r', linestyle='-', linewidth=1)

            ax0.set_title(f"Density Plot for Activity {label} ({activities[label]})")
            ax0.set_xlabel("Value")
            ax0.set_ylabel("Density")
            ax0.legend()

            fig0.savefig(os.path.join(config.results_dir, "density", f"Figure_{label:03}.png"), bbox_inches="tight")
            fig_grid.savefig(os.path.join(config.results_dir, "density", "grid", f"Figure_{label:03}_grid.png"), bbox_inches="tight")

        plt.close('all')