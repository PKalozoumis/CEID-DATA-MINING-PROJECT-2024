import pandas as pd
from matplotlib import pyplot as plt
import os
import seaborn as sns
import numpy as np
import config
import re

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

    if os.path.exists(config.results_dir):
        shutil.rmtree(config.results_dir)

    os.makedirs(config.results_dir)

    li = []

    for file in ["S006.csv"]:
    #for file in os.listdir(config.dataset_dir):
        print(f"Reading {file}...")

        df = pd.read_csv(os.path.join(config.dataset_dir, file), index_col=None, header=0, usecols=["timestamp", "back_x", "back_y", "back_z", "thigh_x", "thigh_y", "thigh_z", "label"])
        
        match = re.match(r"S0([\d]{2})\.csv", file)
        df["participant"] = match.group(1)
        li.append(df)

    df = pd.concat(li, axis=0, ignore_index=True)

    labels = np.sort(df["label"].unique())

    data = df[["back_x", "back_y", "back_z", "thigh_x", "thigh_y", "thigh_z"]].astype("float64")
    data["label"] = df["label"]

    mean = data.groupby("label").mean()
    mean.to_excel(os.path.join(config.results_dir, "mean.xlsx"), index_label="Label")
    print("Saving mean.xlsx...")

    corr = data.groupby("label").corr()
    corr.index.set_names(["label", "dim"], inplace=True)

    for label in labels:
        corr.loc[label].to_excel(os.path.join(config.results_dir, f"corr_{label:03}.xlsx"))
        print(f"Saving corr_{label:03}.xlsx...")

    std = data.groupby("label").std()
    mean.to_excel(os.path.join(config.results_dir, "std.xlsx"), index_label="Label")
    print("Saving std.xlsx...")

    #quantile = df[["back_x", "back_y", "back_z", "thigh_x", "thigh_y", "thigh_z"]].quantile([0.25, 0.5, 0.75])

    #print("Mean\n==================================================")
    #print(mean)

    #print("\nCorrelation\n==================================================")
    #print(corr)

    #print("\nStandard Deviation\n==================================================")
    #print(std)

    #print("\nQuantiles\n==================================================")
    #print(quantile)

    print(mean.index.to_list())


    #Clearing outliers
    #==========================================================================================

    for act in activity_list:
        for attr in attr_list:

            if act not in mean.index.to_list():
                continue

            q1 = data.loc[data["label"] == act][attr].quantile(0.25)
            q3 = data.loc[data["label"] == act][attr].quantile(0.75)
            iqr = q3 - q1

            low = q1 - 1.5*iqr
            high = q3 + 1.5*iqr
            full_count = len(data.loc[data["label"] == act])

            data.loc[data["label"] == act] = data.loc[(data["label"] == act) & (data[attr] > low) & (data[attr] < high)]
            filtered_count =  len(data.loc[data["label"] == act])
            print(f'\nWithin [{low:.3f},{high:.3f}]: {filtered_count} out of {full_count} ({(filtered_count/full_count)*100:.2f} %)\n')
    #==========================================================================================

    print()

    for label in np.sort(df["label"].unique()):

        print(f"Generating plot for Activity {label:03}...")

        current_label_data = data.loc[data["label"] == label]

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16,9))

        for name, col in current_label_data.items():
            sns.kdeplot(col, fill=True, label=name, ax = ax[1])

        ax[1].set_title(f"Density Plot for Activity {label} ({activities[str(label)]})")
        ax[1].set_xlabel("Value")
        ax[1].set_ylabel("Density")
        ax[1].legend()
        #plt.show()

        fig.savefig(os.path.join(config.results_dir, f"Figure_{label:03}.png"), bbox_inches="tight")
        fig.clf()

        #Scatter plot

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