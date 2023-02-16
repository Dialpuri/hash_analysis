"U-Net - Paul Bond"

import tensorflow as tf
from functools import partial

_ARGS = {"padding": "same", "activation": "relu", "kernel_initializer": "he_normal"}
_downsampling_args = {
    "padding": "same",
    "use_bias": False,
    "kernel_size": 3,
    "strides": 1,
}


def model():
    inputs = x = tf.keras.Input(shape=(32, 32, 32, 1))
    skip_list = []
    for filters in [64, 96, 144, 216, 324]:
        x = tf.keras.layers.Conv3D(filters, **_downsampling_args)(x)
        x = tf.keras.layers.BatchNormalization(axis=-1)(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv3D(filters, **_downsampling_args)(x)
        x = tf.keras.layers.BatchNormalization(axis=-1)(x)
        x = tf.keras.layers.ReLU()(x)
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
        x = tf.keras.layers.BatchNormalization(axis=-1)(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Conv3D(filters, 3, **_ARGS)(x)
        x = tf.keras.layers.BatchNormalization(axis=-1)(x)
        x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Conv3D(1, 3, strides=1, use_bias=False, padding="same")(x)
    outputs = tf.keras.activations.tanh(x)

    return tf.keras.Model(inputs, outputs)


if __name__ == "__main__":
    model().summary()
