import tensorflow as tf


def FCN_model(len_classes=5, dropout_rate=0.2):
    # Input layer
    input = tf.keras.layers.Input(shape=(18,))

    # A convolution block
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1)(input)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
