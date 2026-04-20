import pandas as pd
import glob


def load_data(path="dane/dane*.txt"):
    data = pd.concat(
        [pd.read_csv(file, sep="\\s+", header=None, names=["X", "y"])
         for file in sorted(glob.glob(path))],
        ignore_index=True
    )
    return data[["X"]].values, data["y"].values
