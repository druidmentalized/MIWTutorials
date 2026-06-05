import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.callbacks import EarlyStopping


def train_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 100,
    batch_size: int = 60,
    patience: int = 4,
    verbose: int = 1,
):
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=verbose,
    )
    return history


def evaluate_regression_model(
    model,
    X_test: np.ndarray,
    y_test_scaled: np.ndarray,
    scaler,
    model_name: str,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    pred_scaled = model.predict(X_test).flatten()

    pred = np.expm1(scaler.inverse_transform(pred_scaled.reshape(-1, 1))).flatten()
    true = np.expm1(scaler.inverse_transform(y_test_scaled.reshape(-1, 1))).flatten()

    mse = mean_squared_error(true, pred)
    mae = mean_absolute_error(true, pred)

    print(f"{model_name} test MSE: {mse:.8f}")
    print(f"{model_name} test MAE: {mae:.8f}")

    return true, pred, mse, mae
