import matplotlib.pyplot as plt
import pandas as pd


def plan_time_series(df: pd.DataFrame, target_col: str = "Close") -> None:
    plt.figure()
    plt.plot(df["Date"], df[target_col], label=target_col)
    plt.title(f"AMZN {target_col} price over time")
    plt.xlabel("Date")
    plt.ylabel("USD")
    plt.yscale("log")
    plt.legend()
    plt.grid(True)
    plt.show()


def plan_trend(df: pd.DataFrame, target_col: str = "Close") -> None:
    ma_20 = df[target_col].rolling(window=20).mean()
    ma_100 = df[target_col].rolling(window=100).mean()

    plt.figure()
    plt.plot(df["Date"], df[target_col], label=target_col, alpha=0.4)
    plt.plot(df["Date"], ma_20, label="20-day moving average", linewidth=2)
    plt.plot(df["Date"], ma_100, label="100-day moving average", linewidth=2)
    plt.title("Trend: AMZN close price and moving averages")
    plt.xlabel("Date")
    plt.ylabel("USD")
    plt.yscale("log")
    plt.legend()
    plt.grid(True)
    plt.show()


def plan_variability(df: pd.DataFrame, target_col: str = "Close") -> None:
    rolling_std = df[target_col].rolling(window=30).std()

    plt.figure()
    plt.plot(df["Date"], rolling_std, label="30-day rolling std", color="orange")
    plt.title("Variability: 30-day rolling standard deviation of close price")
    plt.xlabel("Date")
    plt.ylabel("Std dev (USD)")
    plt.yscale("log")
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"Mean volatility: {rolling_std.mean():.4f}")
    peak_idx = rolling_std.idxmax()
    print(f"Peak volatility: {df['Date'].iloc[peak_idx].date()}, value: {rolling_std.max():.4f}")


def plan_possible_seasonality(df: pd.DataFrame, target_col: str = "Close") -> None:
    month = df["Date"].dt.month
    monthly_mean = df.groupby(month)[target_col].mean()
    monthly_std = df.groupby(month)[target_col].std()

    plt.figure()
    monthly_mean.plot(kind="bar")
    plt.title("Possible seasonality: average close price by month")
    plt.xlabel("Month")
    plt.ylabel("Average close price (USD)")
    plt.grid(True)
    plt.show()

    print("Monthly standard deviation:")
    print(monthly_std.to_frame(name="Standard deviation"))

    diff = monthly_mean.max() - monthly_mean.min()
    print(f"Highest monthly average: month {monthly_mean.idxmax()}, value {monthly_mean.max():.4f}")
    print(f"Lowest monthly average: month {monthly_mean.idxmin()}, value {monthly_mean.min():.4f}")
    print(f"Difference: {diff:.4f}")


def plot_split(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    target_col: str = "Close",
) -> None:
    plt.figure()
    plt.plot(df_train["Date"], df_train[target_col], label="train")
    plt.plot(df_val["Date"], df_val[target_col], label="validation")
    plt.plot(df_test["Date"], df_test[target_col], label="test")
    plt.title("Chronological data split")
    plt.xlabel("Date")
    plt.ylabel(target_col)
    plt.yscale("log")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_training_history(history, title: str) -> None:
    plt.figure()
    plt.plot(history.history["loss"], label="train loss")
    plt.plot(history.history["val_loss"], label="validation loss")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.yscale("log")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_predictions(dates, true, pred, title: str, pred_label: str) -> None:
    plt.figure()
    plt.plot(dates, true, label="Actual")
    plt.plot(dates, pred, label=pred_label)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Close price (USD)")
    plt.legend()
    plt.grid(True)
    plt.show()
