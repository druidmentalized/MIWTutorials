import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import mean_squared_error, r2_score


def plot_regression_results(
    model,
    X_train, y_train,
    X_test, y_test,
    title="Neural Network Regression",
    save=False
):
    # Predictions
    output_train = model.forward(X_train)
    output_test = model.forward(X_test)

    # Metrics
    final_mse = model.history_mse[-1]
    final_r2 = model.history_r2[-1]

    # Plot setup
    plt.rcParams.update({'font.size': 14})
    plt.figure(figsize=(15, 10))

    # Subplot 1 - Loss over epochs
    plt.subplot(2, 2, 1)
    plt.scatter(range(len(model.history_mse)), model.history_mse, marker=".")
    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.title(f"Loss vs. Epochs, Final MSE: {round(final_mse, 3)}")
    plt.ylim(0, 0.2)

    # Subplot 2 - R2
    plt.subplot(2, 2, 2)
    plt.scatter(range(len(model.history_r2)), model.history_r2, marker=".")
    plt.xlabel("Epochs")
    plt.ylabel("R^2")
    plt.title(f"R^2 vs. Epochs, Final R^2:{round(final_r2, 2)}")
    plt.ylim(0, 1)

    # Subplot 3 - Predictions vs actual data
    plt.subplot(2, 2, 3)
    plt.scatter(X_train, y_train, color='blue', alpha=0.4, label='Train Data')
    plt.scatter(X_train, output_train, color='green', marker='s', label='Train Predictions')
    plt.scatter(X_test, y_test, color='red', alpha=0.4, label='Test Data')
    plt.scatter(X_test, output_test, color='green', marker='x', label='Test Predictions')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Predictions vs Data')
    plt.figtext(0.5, 0.90, f"Neurons={model.hidden_size}, Epochs={len(model.history_r2)}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.legend()


    if save:
        now = datetime.now()
        filename = now.strftime('regression_%Y%m%d_%H%M%S.png')
        plt.savefig(filename)
        print(f"Saved as {filename}")

    plt.show()
