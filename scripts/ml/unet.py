"U-Net - Paul Bond"

import tensorflow as tf

_ARGS = {"padding": "same", "activation": "relu", "kernel_initializer": "he_normal"}


def model():
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



if __name__ == "__main__":
    model().summary()
