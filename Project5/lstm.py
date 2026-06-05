from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


def build_lstm(
    window: int,
    n_features: int,
    units: int = 32,
    dropout: float = 0.2,
) -> Sequential:
    model = Sequential([
        Input(shape=(window, n_features)),
        LSTM(units, activation="tanh"),
        Dropout(dropout),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model
