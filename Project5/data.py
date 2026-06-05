import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

REQUIRED_COLUMNS = ["Date", "Open", "High", "Low", "Close"]
PRICE_COLUMNS = ["Close", "Open", "High", "Low"]


def load_split_data(
    path: str,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> tuple[pd.DataFrame,
           tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
           MinMaxScaler]:
    df = _load_clean(path)
    df_train, df_val, df_test = _chronological_split(df, train_frac, val_frac)

    # Fit on all data (log1p domain) so that test prices, which are much
    # higher than training prices on a long-running stock like AMZN, still
    # land inside [0, 1] for the model.
    scaler_y = MinMaxScaler()
    scaler_y.fit(np.log1p(df[["Close"]].values))

    return df, (df_train, df_val, df_test), scaler_y


def _load_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available: {list(df.columns)}")

    df = df[REQUIRED_COLUMNS].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").dropna().reset_index(drop=True)
    return df


def _chronological_split(
    df: pd.DataFrame,
    train_frac: float,
    val_frac: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    train_end = int(train_frac * n)
    val_end = int((train_frac + val_frac) * n)

    df_train = df.iloc[:train_end].copy()
    df_val = df.iloc[train_end:val_end].copy()
    df_test = df.iloc[val_end:].copy()
    return df_train, df_val, df_test


def scale_y_log(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    scaler: MinMaxScaler,
    target_col: str = "Close",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_y = scaler.transform(np.log1p(df_train[[target_col]].values)).flatten()
    val_y = scaler.transform(np.log1p(df_val[[target_col]].values)).flatten()
    test_y = scaler.transform(np.log1p(df_test[[target_col]].values)).flatten()
    return train_y, val_y, test_y


def create_sequences_1d(series: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(window, len(series)):
        X.append(series[i - window:i])
        y.append(series[i])
    X = np.array(X).reshape((-1, window, 1))
    y = np.array(y)
    return X, y


def create_sequences_multifeature(
    X_data: np.ndarray,
    y_data: np.ndarray,
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
    X_seq, y_seq = [], []
    for i in range(window, len(X_data)):
        X_seq.append(X_data[i - window:i, :])
        y_seq.append(y_data[i])
    return np.array(X_seq), np.array(y_seq)


def split_sequences_by_target_index(
    X_seq: np.ndarray,
    y_seq: np.ndarray,
    target_indices: np.ndarray,
    train_end_idx: int,
    val_end_idx: int,
) -> tuple[np.ndarray, np.ndarray,
           np.ndarray, np.ndarray,
           np.ndarray, np.ndarray,
           np.ndarray]:
    train_mask = target_indices < train_end_idx
    val_mask = (target_indices >= train_end_idx) & (target_indices < val_end_idx)
    test_mask = target_indices >= val_end_idx

    return (
        X_seq[train_mask], y_seq[train_mask],
        X_seq[val_mask], y_seq[val_mask],
        X_seq[test_mask], y_seq[test_mask],
        test_mask,
    )


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    df_features = df.copy()
    df_features["Range"] = df_features["High"] - df_features["Low"]
    df_features = df_features.dropna().reset_index(drop=True)
    return df_features


def scale_multi_features(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    df_all: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "Close",
) -> tuple[np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray,
           MinMaxScaler, MinMaxScaler]:
    log_indices = [feature_cols.index(c) for c in PRICE_COLUMNS if c in feature_cols]

    def to_log(values: np.ndarray) -> np.ndarray:
        values = values.copy()
        values[:, log_indices] = np.log1p(values[:, log_indices])
        return values

    scaler_y = MinMaxScaler()
    scaler_y.fit(np.log1p(df_all[[target_col]].values))

    scaler_X = MinMaxScaler()
    scaler_X.fit(to_log(df_all[feature_cols].values))

    X_train = scaler_X.transform(to_log(df_train[feature_cols].values))
    X_val = scaler_X.transform(to_log(df_val[feature_cols].values))
    X_test = scaler_X.transform(to_log(df_test[feature_cols].values))

    y_train = scaler_y.transform(np.log1p(df_train[[target_col]].values)).flatten()
    y_val = scaler_y.transform(np.log1p(df_val[[target_col]].values)).flatten()
    y_test = scaler_y.transform(np.log1p(df_test[[target_col]].values)).flatten()

    # Keep X[Close] and y on identical scale (Homework cell 21 invariant).
    close_idx = feature_cols.index(target_col)
    X_train[:, close_idx] = y_train
    X_val[:, close_idx] = y_val
    X_test[:, close_idx] = y_test

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler_X, scaler_y
