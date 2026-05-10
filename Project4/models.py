from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, Activation
from keras.models import Model

NUM_CLASSES = 10
INPUT_SHAPE = (32, 32, 3)
FILTERS_START = 32
DROPOUT_START = 0.5
DROPOUT_STEP = 0.05


def build_model(blocks_count: int, batch_norm: bool = False) -> Model:
    inputs = Input(shape=INPUT_SHAPE)

    x = inputs
    for i in range(blocks_count):
        filters = FILTERS_START * (2 ** i)
        dropout = DROPOUT_START - (DROPOUT_STEP * i)

        x = Conv2D(filters, kernel_size=(3, 3), padding='same')(x)
        if batch_norm: x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(filters, kernel_size=(3, 3), padding='same')(x)
        if batch_norm: x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(dropout)(x)

    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.25)(x)
    outputs = Dense(NUM_CLASSES, activation='softmax')(x)

    return Model(inputs=inputs, outputs=outputs)


def build_model_a() -> Model:
    return build_model(blocks_count=1)


def build_model_b(blocks_count=3) -> Model:
    return build_model(blocks_count=blocks_count)


def build_model_c(blocks_count=5) -> Model:
    return build_model(blocks_count=blocks_count, batch_norm=True)
