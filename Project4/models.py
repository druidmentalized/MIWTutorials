from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, Activation
from keras.models import Model

NUM_CLASSES = 10
INPUT_SHAPE = (32, 32, 3)
FILTERS_START = 32
DROPOUT_START = 0.3
DROPOUT_STEP = 0.05
EPOCHS = 45
BATCH_SIZE = 128


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


def train_model(model: Model, dataset, x_val, y_val):
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
        ModelCheckpoint(filepath=f'{model.name}_best.keras', monitor='val_loss', save_best_only=True),
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
    ]
    return model.fit(
        dataset,
        epochs=EPOCHS,
        verbose='auto',
        validation_data=(x_val, y_val),
        callbacks=callbacks,
    )


def build_model_a() -> Model:
    return build_model(blocks_count=1)


def build_model_b(blocks_count=3) -> Model:
    return build_model(blocks_count=blocks_count)


def build_model_c(blocks_count=5) -> Model:
    return build_model(blocks_count=blocks_count, batch_norm=True)
