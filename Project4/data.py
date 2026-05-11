import tensorflow as tf
from keras import layers
from keras.datasets.cifar10 import load_data as load_cifar10
from keras.utils import to_categorical
import numpy as np
from sklearn.model_selection import train_test_split


def load_data(rs: int):
    (x_train_all, y_train_all), (x_test_all, y_test_all) = load_cifar10()

    x_all = np.concatenate((x_train_all, x_test_all))
    y_all = np.concatenate((y_train_all, y_test_all))

    x_train, x_temp, y_train, y_temp = train_test_split(
        x_all,
        y_all,
        test_size=0.6,
        random_state=rs
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.5,
        random_state=rs
    )

    x_train = x_train.astype('float32') / 255
    x_val = x_val.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    y_train = to_categorical(y_train, 10)
    y_val = to_categorical(y_val, 10)
    y_test = to_categorical(y_test, 10)

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def get_dataset(x_train, y_train, batch_size: int = 64) -> tf.data.Dataset:
    augmentation = tf.keras.Sequential([
        layers.RandomFlip('horizontal'),
        layers.RandomRotation(15 / 360),
        layers.RandomTranslation(0.1, 0.1),
    ])
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    dataset = dataset.shuffle(len(x_train), reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(
        lambda x, y: (augmentation(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    return dataset.prefetch(tf.data.AUTOTUNE)
