import os
import numpy as np
import tensorflow as tf
import pandas as pd
import random
import math


def extract_data_from_file(file_path):
    headers = ["u", "v", "w", "data", "type"]
    df = pd.read_csv(file_path, names=headers)

    df = df.replace("X", 0)
    df = df.replace("C", 1)
    df = df.replace("P", 1)
    df = df.replace("O", 1)

    data = np.array(df["data"]).reshape(32, 32, 32, 1)
    labels = np.array(df["type"]).reshape(32, 32, 32, 1)

    return data, labels


def import_data(target_dir_1: str, target_dir_2):
    print("Loading dataset...")
    dir_1_data = []
    dir_2_data = []

    dir_1_total = 0
    for file in os.scandir(target_dir_1):
        dir_1_data_label_tuple = extract_data_from_file(file.path)
        dir_1_data.append(dir_1_data_label_tuple)
        dir_1_total += 1

    for index, file in enumerate(os.scandir(target_dir_2)):
        if index >= dir_1_total:
            break
        dir_2_data_label_tuple = extract_data_from_file(file.path)
        dir_2_data.append(dir_2_data_label_tuple)

    data = dir_1_data + dir_2_data
    random.shuffle(data)
    print("Loading dataset complete")
    return data


def test_train_split(dataset, split=0.2):
    print("Splitting data")
    training_index = math.floor((1 - split) * len(dataset))
    training_data = dataset[:training_index]
    test_data = dataset[training_index:]

    # print(f"Splitting data complete. Training data size : {len(training_data)}, Testing data size : {len(test_data)}")
    return training_data, test_data


def _model():
    _ARGS = {"padding": "same", "activation": "relu", "kernel_initializer": "he_normal"}

    inputs = x = tf.keras.Input(shape=(32, 32, 32, 1))

    skip_list = []
    for filters in [64, 96, 144, 216, 324]:
        x = tf.keras.layers.Conv3D(filters, 3, **_ARGS)(x)
        x = tf.keras.layers.Conv3D(filters, 3, **_ARGS)(x)
        skip_list.append(x)
        x = tf.keras.layers.MaxPool3D(2)(x)
        x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Conv3D(486, 3, **_ARGS)(x)
    x = tf.keras.layers.Conv3D(486, 3, **_ARGS)(x)

    for filters in [324, 216, 144, 96, 64]:
        x = tf.keras.layers.Conv3DTranspose(filters, 3, 2, padding="same")(x)
        x = tf.keras.layers.concatenate([x, skip_list.pop()])
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Conv3D(filters, 3, **_ARGS)(x)
        x = tf.keras.layers.Conv3D(filters, 3, **_ARGS)(x)
    outputs = tf.keras.layers.Conv3D(1, 3, padding="same")(x)
    return tf.keras.Model(inputs, outputs)


def main():
    dir_1 = "./debug/32x32x32_points/sugars"
    dir_2 = "./debug/32x32x32_points/non_sugars"
    dataset = import_data(dir_1, dir_2)
    train, test = test_train_split(dataset, split=.2)

    model = _model()
    model.summary()

    model.compile(optimizer="adam", loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])

    train_data = (x for x in train)
    test_labels = (x for x in test)

    spec = tf.TensorSpec(shape=(32, 32, 32, 1), dtype=tf.float32)

    batch_size = 1
    train_dataset = tf.data.Dataset.from_generator(lambda: train_data, output_signature=(spec, spec)).batch(
        batch_size=batch_size)
    test_dataset = tf.data.Dataset.from_generator(lambda: test_labels, output_signature=(spec, spec)).batch(
        batch_size=batch_size)

    model.fit(
        x=train_dataset,
        epochs=20,
        steps_per_epoch=10,
        validation_data=test_dataset,
        validation_steps=100,
        verbose=2,
    )




if __name__ == "__main__":
    main()
