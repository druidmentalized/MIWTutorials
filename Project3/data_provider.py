import pandas as pd
import glob

from sklearn.preprocessing import MinMaxScaler


def load_data(path="dane/dane*.txt"):
    data = pd.concat(
        [pd.read_csv(file, sep="\\s+", header=None, names=["X", "y"])
         for file in sorted(glob.glob(path))],
        ignore_index=True
    )
    return data[["X"]].values, data["y"].values

def normalize_data(split_data):
    X_train, X_test, y_train, y_test = split_data

    scaler_X = MinMaxScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)

    scaler_y = MinMaxScaler()
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

    return X_train, X_test, y_train, y_test
