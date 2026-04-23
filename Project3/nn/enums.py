from enum import Enum, auto


class UpdateMode(Enum):
    BATCH = auto()
    STOCHASTIC = auto()
