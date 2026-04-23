from enum import Enum, auto

class Activation(Enum):
    RELU = auto()
    TANH = auto()

class WeightInit(Enum):
    STANDARD = auto()
    HE = auto()

class UpdateMode(Enum):
    BATCH = auto()
    STOCHASTIC = auto()
