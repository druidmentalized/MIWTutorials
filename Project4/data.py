from keras.datasets.cifar10 import load_data as load_cifar10
from keras.src.legacy.preprocessing.image import ImageDataGenerator
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


def get_augmentor(x_train):
    augmentor = ImageDataGenerator(
        rotation_range=15,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1
    )
    augmentor.fit(x_train)
    return augmentor
