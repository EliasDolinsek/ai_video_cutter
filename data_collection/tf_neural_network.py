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


def transform_df_data(df):
    return (df["cut"], df.drop("cut", axis=1))


def create_model():
    return keras.Sequential([
        keras.layers.Flatten(input_shape=(68,)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(2, activation='softmax')
    ])


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        folder = "data/"
    else:
        folder = sys.argv[1]

    data = load_data(folder)

    y_data = []
    x_data = []

    # Prepare data for tf
    for df in data:
        y,x = transform_df_data(df)

        y_data.append(y)
        x_data.append(x)

    
    model = create_model()
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])


    checkpoint_path = "training_1/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)

    # Train data
    data_len = len(x_data)
    for x_data, y_data, index in zip(x_data, y_data, range(data_len)):
        print(f"\nTraining {index}/{data_len}")
        model.fit(x_data, y_data, epochs=100, callbacks=[cp_callback], verbose=1)

    print("\nFinished training")
    model.save("model.h5")
