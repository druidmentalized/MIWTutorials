import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score


def plot_decision_boundary_with_metrics(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    title="Model",
    show_precision=False
):
    h = 0.02

    # Define grid
    x_min = min(X_train[:, 0].min(), X_test[:, 0].min()) - 1
    x_max = max(X_train[:, 0].max(), X_test[:, 0].max()) + 1
    y_min = min(X_train[:, 1].min(), X_test[:, 1].min()) - 1
    y_max = max(X_train[:, 1].max(), X_test[:, 1].max()) + 1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h)
    )

    # Predict grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Predictions on real data
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    if show_precision:
        precision = precision_score(y_test, y_test_pred)
        metrics_text = f"Train={train_acc:.2f} | Test={test_acc:.2f} | Prec={precision:.2f}"
    else:
        metrics_text = f"Train={train_acc:.2f} | Test={test_acc:.2f}"

    # Plot
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, marker='o', label="Train")
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker='x', label="Test")

    plt.title(f"{title}\n{metrics_text}")
    plt.legend()
    plt.show()
