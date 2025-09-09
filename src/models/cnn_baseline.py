
import tensorflow as tf
from tensorflow.keras import layers, models

def build_cnn(input_shape, n_classes):
    x_in = layers.Input(shape=input_shape)
    x = layers.Conv2D(16, (2,2), padding="same", activation="relu")(x_in)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(32, (3,3), activation="relu")(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(64, (5,5), activation="relu")(x)
    x = layers.MaxPooling2D((5,5))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    return models.Model(x_in, out)
