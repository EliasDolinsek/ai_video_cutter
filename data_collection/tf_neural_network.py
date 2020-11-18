from sklearn import model_selection
from sklearn.utils import shuffle

import os, sys
import tensorflow as tf
import keras as keras
import pandas as pd
import matplotlib.pyplot as plt


def load_data(folder):
    for (dirpath, dirnames, filenames) in os.walk(folder):
        files = [os.path.join(dirpath, filename) for filename in filenames]

    data = []
    for file in files:
        csv_data = pd.read_csv(file, index_col=0, header=0)

        # Delete timestamps
        del csv_data["start_timestamp"]
        del csv_data["end_timestamp"]

        # Remove white spaces
        csv_data.columns = csv_data.columns.str.replace(" ", "")
        data.append(csv_data)

    return data


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        folder = "data/"
    else:
        folder = sys.argv[1]

    data = load_data(folder)
    print(len(data))
