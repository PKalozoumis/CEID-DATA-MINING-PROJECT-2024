import pandas as pd
import os
import config
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

from sklearn.model_selection import train_test_split

#====================================================================================================

if __name__ == "__main__":

    df = pd.read_csv("data.csv", header=0)

    X = df.drop(columns="label")
    Y = df[["label"]]

    X_train, X_test, y_train, y_ytest = train_test_split(X, Y, test_size=0.3, random_state=1997)

    model = Sequential()

    model.add(Dense(32, activation="relu", input_dim=30))
    model.add(Dense(12, activation="softmax", input_dim=32))

    optimizer = SGD(learning_rate=0.001)

    model.compile(loss=)

    print(df)