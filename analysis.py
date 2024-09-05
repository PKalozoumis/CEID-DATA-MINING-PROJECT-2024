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

import argparse

import shutil

activities = {
    "1": "walking",
    "2": "running",
    "3": "shuffling",
    "4": "stairs ascending",
    "5": "stairs descending",
    "6": "standing",
    "7": "sitting",
    "8": "lying",
    "13": "cycling sit",
    "14": "cycling stand",
    "130": "cycling sit inactive",
    "140": "cycling stand inactive"
}

attr_list = ["back_x", "back_y", "back_z", "thigh_x", "thigh_y", "thigh_z"]
activity_list = [1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 130, 140]

# noinspection SpellCheckingInspection
if __name__ == "__main__":

    #Arguments
    #===================================================================================================================
    parser = argparse.ArgumentParser(description='Dataset Analysis', allow_abbrev=False)
    parser.add_argument('--no-outliers', action="store_true", default=False, help="Remove outliers from the dataset")
    args = parser.parse_args()
    
    #Config and directory initialization
    #===================================================================================================================
    config = None

    with open("config.json", "r") as f:
        temp = json.load(f)
        Config = namedtuple("Config", temp.keys())
        config = Config(**temp)

    if os.path.exists(config.results_dir):
        shutil.rmtree(config.results_dir)

    if os.path.exists("proj/no_outliers"):
        shutil.rmtree("proj/no_outliers")

    os.makedirs(config.results_dir)
    os.makedirs(f"{config.results_dir}/outliers")
    os.makedirs(f"{config.results_dir}/density")
    os.makedirs(f"{config.results_dir}/timeseries")
    os.makedirs(f"{config.results_dir}/heatmaps")
    os.makedirs(f"{config.results_dir}/density/grid")
    os.makedirs("proj/no_outliers")

    #Read files into one dataframe
    #===================================================================================================================
    li = []

    for file in ["S006.csv"]:
    #for file in os.listdir(config.dataset_dir):
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
    mean.to_excel(os.path.join(config.results_dir, "mean.xlsx"), index_label="Label")
    print("Saving mean.xlsx...")

    median = data.groupby("label").median()
    median.to_excel(os.path.join(config.results_dir, "median.xlsx"), index_label="Label")
    print("Saving median.xlsx...\n")

    corr = data.groupby("label").corr()
    corr.index.set_names(["label", "dim"], inplace=True)

    for label in labels:
        corr.loc[label].to_excel(os.path.join(config.results_dir, f"corr_{label:03}.xlsx"))
        print(f"Saving corr_{label:03}.xlsx...\n")

    std = data.groupby("label").std()
    mean.to_excel(os.path.join(config.results_dir, "std.xlsx"), index_label="Label")
    print("Saving std.xlsx...\n")

    #Clearing outliers
    #===================================================================================================================

    no_outliers = None

    if args.no_outliers:

        no_outliers = copy.deepcopy(df)

        total_outliers = 0

        for act in activity_list:
            
            if act not in median.index.to_list():
                    continue
            
            for attr in attr_list:

                '''
                abs_deviation = np.abs(no_outliers.loc[no_outliers["label"] == act, attr] - median.loc[act, attr])
                q3 = no_outliers.loc[no_outliers["label"] == act][attr].quantile(0.75)

                mad = abs_deviation.median()

                print(mad)

                #abs_deviation = no_outliers - median

                low = median.loc[act, attr] - 3*mad
                high = median.loc[act, attr] + 3*mad
                '''

                low = mean.loc[act, attr] - 3*std.loc[act, attr]
                high = mean.loc[act, attr] + 3*std.loc[act, attr]

                full_count = len(no_outliers.loc[no_outliers["label"] == act])
                no_outliers = no_outliers.loc[(no_outliers["label"] != act) | ((no_outliers["label"] == act) & (no_outliers[attr] > low) & (no_outliers[attr] < high))]
                filtered_count =  len(no_outliers.loc[no_outliers["label"] == act])

                total_outliers += full_count - filtered_count

                print(f'Within [{low:.3f},{high:.3f}]: {filtered_count} out of {full_count} ({(filtered_count/full_count)*100:.2f} %)')

        print(f"\nRemoved a total of {total_outliers} outliers")

        #Rewrite the files, without outliers
        for participant in no_outliers["participant"].unique():
            print(f"Writing S0{participant:02}_no_outliers.csv...")
            no_outliers.loc[no_outliers["participant"] == participant].drop("participant", axis=1).to_csv("proj/no_outliers/"f"S0{participant:02}_no_outliers.csv", index=False)

    #print(no_outliers)

    #mean = no_outliers.groupby("label").mean()
    #mean.to_excel(os.path.join(config.results_dir, "mean_no_outliers.xlsx"), index_label="Label")
    #print("Saving mean_no_outliers.xlsx...")

    #Time Series Plots
    #===================================================================================================================

    for participant in df["participant"].unique():

        pdata = df.loc[df["participant"] == participant]

        #Split participant data based on consecutive same value on the label column
        #This will show us how many times the participant changed activity

        s = pdata["label"].ne(pdata["label"].shift()).cumsum()
        groups = [(pdata.iloc[0]["label"], pdata) for _, pdata in pdata.groupby(s)]

        #Initialize plot
        rows = math.floor(math.sqrt(len(groups)))
        cols = math.ceil(len(groups)/rows)
        tfig, tax = plt.subplots(nrows=rows, ncols=cols, figsize=(16,9))

        #print(groups)

        for i, (label, df2) in enumerate(groups):

            subplot = tax[i // rows, i % rows]

            for attr in ["back_x", "back_y", "back_z", "thigh_x", "thigh_y", "thigh_z"]:
                subplot.plot(df2["timestamp"], df2[attr], label=attr)

            subplot.legend()
            subplot.set_xlabel("Timestamp")
            subplot.set_ylabel("Value")
            subplot.set_title(f'{label}')

        tfig.suptitle(f"Time series data for participant {participant}")

        tfig.savefig(os.path.join(config.results_dir, "timeseries", f"S0{participant:02}_tseries.png"))

    sys.exit()

    #Plots
    #===================================================================================================================

    for label in np.sort(df["label"].unique()):

        print(f"Generating plots for Activity {label:03}...")

        #Correlation heatmap
        #-----------------------------------------------------------------------------------------------------------------------
        fig, ax = plt.subplots()
        sns.heatmap(corr.loc[label], annot=True, cmap='coolwarm', fmt='.3f', ax = ax, vmin=-1, vmax=1)
        ax.set_title(f"Correlation heatmap for Activity {label} ({activities[str(label)]})")
        fig.savefig(os.path.join(config.results_dir, "heatmaps", f"corr_{label:03}.png"))

        #Density plots
        #-----------------------------------------------------------------------------------------------------------------------
        current_label_data = data.loc[data["label"] == label, ["back_x", "back_y", "back_z", "thigh_x", "thigh_y", "thigh_z"]]
        #current_label_data_no_outliers = data.loc[no_outliers["label"] == label, ["back_x", "back_y", "back_z", "thigh_x", "thigh_y", "thigh_z"]]

        fig0, ax0 = plt.subplots()
        fig_grid, ax_grid = plt.subplots(nrows = 2, ncols = 3, figsize=(16,9))
        #fig1, ax1 = plt.subplots(nrows=1, ncols=2, figsize=(16,9))

        for i, (name, col) in enumerate(current_label_data.items()):
            sns.kdeplot(col, fill=True, label=name, ax = ax0)
            sns.kdeplot(col ,fill=True, label=name, ax = ax_grid[i//3, i%3])
            #sns.kdeplot(col, fill=True, label=name, ax = ax1[0])

            ax_grid[i//3, i%3].set_title(f"Density Plot of {name} for Activity {label} ({activities[str(label)]})")
            ax_grid[i//3, i%3].set_xlabel("Value")
            ax_grid[i//3, i%3].set_ylabel("Density")

            low = median.loc[label, name] - 3*std.loc[label, name]
            high = median.loc[label, name] + 3*std.loc[label, name]

            ax_grid[i//3, i%3].axvline(x=low, color='r', linestyle='-', linewidth=1)
            ax_grid[i//3, i%3].axvline(x=high, color='r', linestyle='-', linewidth=1)

        '''
        for name, col in current_label_data_no_outliers.items():
            sns.kdeplot(col, fill=True, label=name, ax = ax1[1])
        '''

        ax0.set_title(f"Density Plot for Activity {label} ({activities[str(label)]})")
        ax0.set_xlabel("Value")
        ax0.set_ylabel("Density")
        ax0.legend()

        '''
        ax1[0].set_title(f"Density Plot for Activity {label} ({activities[str(label)]})")
        ax1[0].set_xlabel("Value")
        ax1[0].set_ylabel("Density")
        ax1[0].legend()

        ax1[1].set_title(f"Density Plot for Activity {label} ({activities[str(label)]}) (no outliers)")
        ax1[1].set_xlabel("Value")
        ax1[1].set_ylabel("Density")
        ax1[1].legend()
        '''

        #plt.show()

        fig0.savefig(os.path.join(config.results_dir, "density", f"Figure_{label:03}.png"), bbox_inches="tight")
        fig_grid.savefig(os.path.join(config.results_dir, "density", "grid", f"Figure_{label:03}_grid.png"), bbox_inches="tight")
        #fig1.savefig(os.path.join(config.results_dir, "outliers", f"Figure_{label:03}_outliers.png"), bbox_inches="tight")
        #Scatter plot

        plt.close('all')

    #sns.lmplot(x="thigh_x", y="back_x", data=current_label_data)
    #plt.show()

    #======================================================================


    '''

    melted = data.loc[data["label"] == 1, ["back_x", "back_y", "back_z", "thigh_x", "thigh_y", "thigh_z"]].melt(var_name="Category", value_name="Value")

    print(f'\nBox plot kept {len(data)} out of {full_count} ({(len(data)/full_count)*100}%)\n')


    fig, ax = plt.subplots()
    melted.boxplot(by="Category", column="Value")

    ax.set_title("Box Plot")
    ax.set_xlabel("Category")
    ax.set_ylabel("Value")

    plt.show()
    '''