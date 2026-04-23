from enum import Enum

import numpy as np


class WeightInit(Enum):
    STANDARD = ("standard", lambda size_in: 1.0)
    HE = ("he", lambda size_in: np.sqrt(2.0 / size_in))

    def __init__(self, name, scaler):
        self.name = name
        self.scaler = scaler

    def initialize(self, rgen, size_in, size_out):
        weights = rgen.standard_normal(size=(size_in, size_out))
        return weights * self.scaler(size_in)
