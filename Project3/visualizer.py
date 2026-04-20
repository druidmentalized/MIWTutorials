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
    output_train = model.predict(X_train)
    output_test = model.predict(X_test)

    # Metrics
    train_mse = mean_squared_error(y_train, output_train)
    test_mse = mean_squared_error(y_test, output_test)
    train_r2 = r2_score(y_train, output_train)
    test_r2 = r2_score(y_test, output_test)

    # Loss curve from MLPRegressor
    loss_history = model.loss_curve_

    # Plot setup
    plt.rcParams.update({'font.size': 14})
    plt.figure(figsize=(15, 10))

    # Subplot 1 - Loss over epochs
    plt.subplot(2, 2, 1)
    plt.plot(loss_history, marker='.')
    plt.ylabel('MSE')
    plt.xlabel('Epochs')
    plt.title(f'Loss vs. Epochs, Final MSE: {round(train_mse, 3)}')
    plt.ylim(0, max(loss_history) * 1.1)

    # Subplot 2 - R2 is scalar so we just display it as text
    plt.subplot(2, 2, 2)
    plt.axis('off')
    plt.text(0.5, 0.5,
             f'Train MSE: {train_mse:.4f}\n'
             f'Test MSE:  {test_mse:.4f}\n\n'
             f'Train R²:  {train_r2:.4f}\n'
             f'Test R²:   {test_r2:.4f}',
             ha='center', va='center',
             fontsize=16,
             transform=plt.gca().transAxes
             )
    plt.title('Final Metrics')

    # Subplot 3 - Predictions vs actual data
    plt.subplot(2, 2, 3)
    plt.scatter(X_train, y_train, color='blue', alpha=0.4, label='Train Data')
    plt.scatter(X_test, y_test, color='red', alpha=0.4, label='Test Data')
    plt.scatter(X_train, output_train, color='green', marker='s', alpha=0.4, label='Train Predictions')
    plt.scatter(X_test, output_test, color='green', marker='x', alpha=0.6, label='Test Predictions')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Predictions vs Data')
    plt.legend()

    # Subplot 4 - Smooth predicted curve over full range
    plt.subplot(2, 2, 4)
    X_range = np.linspace(X_train.min(), X_test.max(), 300).reshape(-1, 1)
    y_curve = model.predict(X_range)
    plt.scatter(X_train, y_train, color='blue', alpha=0.4, label='Train Data')
    plt.scatter(X_test, y_test, color='red', alpha=0.4, label='Test Data')
    plt.plot(X_range, y_curve, color='green', linewidth=2, label='Model Curve')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Fitted Curve')
    plt.legend()

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()

    # Save if requested
    if save:
        now = datetime.now()
        filename = now.strftime(f'regression_%Y%m%d_%H%M%S.png')
        plt.savefig(filename)
        print(f"Saved as {filename}")

    plt.show()
