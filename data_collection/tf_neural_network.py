import os
import sys

import keras as keras
import pandas as pd
import tensorflow as tf


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
    return df["cut"], df.drop("cut", axis=1)


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

    y_data_list = []
    x_data_list = []

    # Prepare data for tf
    for df in data:
        y, x = transform_df_data(df)

        y_data_list.append(y)
        x_data_list.append(x)

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
    data_len = len(x_data_list)
    for x_data_list, y_data_list, index in zip(x_data_list, y_data_list, range(data_len)):
        print(f"\nTraining content {index}/{data_len}")

        x_data_values = x_data_list.values
        x_data_values_len = len(x_data_values)
        for x_data, y_data, time_frame_index in zip(x_data_values, y_data_list.values, range(x_data_values_len)):
            print(f"Time frame {time_frame_index}/{x_data_values_len}")
            model.fit(x_data_list, y_data_list, epochs=1, callbacks=[cp_callback], verbose=1)

    print("\nFinished training")
    model.save("model.h5")
