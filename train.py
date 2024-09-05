import pandas as pd
import os
import numpy as np
import json
import sys
from collections import namedtuple

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split

pd.options.mode.chained_assignment = None  # default='warn'

#====================================================================================================

if __name__ == "__main__":

    config = None
    num_epochs = 80

    with open("config.json", "r") as f:
        temp = json.load(f)
        Config = namedtuple("Config", temp.keys())
        config = Config(**temp)


    df = pd.read_csv(os.path.join(config.train_dir, "W_S006.csv"), header=0)

    X = df.drop(columns=["label", "participant"])
    Y = df[["label"]]
    
    labels = [1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 130, 140]

    for label in labels:
        Y["out" + str(label)] = (Y["label"] == label).apply(int)

    Y.drop("label", inplace=True, axis=1)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1997)

    model = Sequential()

    model.add(Dense(32, activation="relu", input_dim=30))
    model.add(Dense(12, activation="softmax", input_dim=32))

    optimizer = SGD(learning_rate=0.001)

    model.compile(loss='categorical_crossentropy', optimizer = optimizer)

    history = model.fit(X_train, Y_train, epochs=80, verbose=0, validation_data=(X_test, Y_test))

    loss_train = np.pad(history.history["loss"], (0, max(0, num_epochs - len(history.history["loss"]))), mode="edge")
    loss_valid = np.pad(history.history["val_loss"], (0, max(0, num_epochs - len(history.history["val_loss"]))), mode="edge")

    plt.plot(loss_train, label=f"Train Loss")
    plt.plot(loss_valid, label=f"Valid Loss")

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Convergence Plot per Training Cycle')
    plt.legend()
    plt.show()

    print(df)