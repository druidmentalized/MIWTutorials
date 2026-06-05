from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout


def build_simple_rnn(
    window: int,
    n_features: int = 1,
    units: int = 32,
    dropout: float = 0.2,
) -> Sequential:
    model = Sequential([
        Input(shape=(window, n_features)),
        SimpleRNN(units, activation="tanh"),
        Dropout(dropout),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model
