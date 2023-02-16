import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import unet

import data_import


def train_MLP():
    sugar_data_set = data_import.extract_data_from_folder("./debug/histogram_data_sugars")
    no_sugar_data_set = data_import.extract_data_from_folder("./debug/histogram_data_no_sugars")

    no_sugar_data_set_cut = no_sugar_data_set.head(len(sugar_data_set))

    combined_df = data_import.combined_datasets_with_labels(sugar_data_set, no_sugar_data_set_cut, shuffle=True)

    combined_df = combined_df[combined_df["Theta_0"] < 100]

    train, test = train_test_split(combined_df, train_size=0.2, shuffle=True)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(8),
        tf.keras.layers.Dense(2),
    ])

    train_labels = train["type"]
    train_values = train.drop("type", axis=1)

    test_labels = test["type"]
    test_values = test.drop("type", axis=1)

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(train_values, train_labels, epochs=1000)
    test_loss, test_acc = model.evaluate(test_values, test_labels, verbose=2)
    print('\nTest accuracy:', test_acc)


def get_data_generator(df: pd.DataFrame):

    print(df)

    data_headers = []
    label_headers = []

    grid_size = 8

    for i in range(0, grid_size, 1):
        for j in range(0, grid_size, 1):
            for k in range(0, grid_size, 1):
                header = f"{i}{j}{k}"
                data = header + "_data"
                sugar = header + "_sugar"
                data_headers.append(data)
                label_headers.append(sugar)

    df = df.replace("X", 0)
    df = df.replace(" C", 1)
    df = df.replace(" P", 1)
    df = df.replace(" O", 1)

    print(np.array(df.columns).reshape(8,8,8,2))

    for index, row in df.iterrows():
        data_values = np.array(row[:512].values).reshape(8, 8, 8).reshape(8,8,8,1)
        label_values = np.array(row[512:].values).reshape(8, 8, 8).reshape(8,8,8,1)

        yield data_values, label_values

def train():
    df_1 = pd.read_csv("./debug/labelled_points/data.csv", index_col=False)
    df_2 = pd.read_csv("./debug/labelled_points/labels.csv", index_col=False)
    df_1.pop(df_1.columns[0])
    df_2.pop(df_2.columns[0])

    combined_df = pd.concat([df_1, df_2], axis=1)

    train, test = train_test_split(combined_df, train_size=0.2, shuffle=True)

    _train_gen = get_data_generator(train)
    _test_gen = get_data_generator(test)

    batch_size = 1

    spec = tf.TensorSpec(shape=(8, 8, 8, 1), dtype=tf.float32)
    train_dataset = tf.data.Dataset.from_generator(lambda: _train_gen, output_signature=(spec, spec)).batch(
        batch_size=batch_size)
    test_dataset = tf.data.Dataset.from_generator(lambda: _test_gen, output_signature=(spec, spec)).batch(
        batch_size=batch_size)

    model = unet.model()
    model.summary()
    model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(
        train_dataset,
        epochs=100,
        steps_per_epoch=10,
        validation_data=test_dataset,
        validation_steps=100,
        verbose=2
    )


if __name__ == "__main__":
    train()
