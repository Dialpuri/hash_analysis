from sklearn.model_selection import train_test_split
import tensorflow as tf
from unet import UNet2D

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
    test_loss, test_acc = model.evaluate(test_values,  test_labels, verbose=2)
    print('\nTest accuracy:', test_acc)


if __name__ == "__main__":
    train_MLP()
