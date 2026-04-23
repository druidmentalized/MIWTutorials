from enum import Enum

import numpy as np


class Activation(Enum):
    RELU = (
        "relu",
        lambda x: np.maximum(0, 1 * x),
        lambda x: np.where(x > 0, 1, np.where(x < 0, 0, 0.5))
    )

    TANH = (
        "tanh",
        lambda x: np.tanh(x),
        lambda x: 1.0 - np.tanh(x) ** 2
    )

    def __init__(self, name, forward, derivative):
        self.name = name
        self.forward = forward
        self.derivative = derivative
