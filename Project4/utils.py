import numpy as np
import pandas as pd
from datetime import datetime

from keras import Model
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def plot_history(history, marker):
    df = pd.DataFrame(history.history)
    ax = df.plot()

    fig = ax.get_figure()
    timestamp = datetime.now().strftime("%d%m%Y")
    fig.savefig(f'history_plot_{marker}_{timestamp}.png')
    plt.show()


def evaluate_model(model: Model, x_test, y_test, marker):
    loss, accuracy = model.evaluate(x_test, y_test, verbose='auto')
    print(f'Test loss of {marker}:', loss)
    print(f'Test accuracy {marker}:', accuracy)


def show_confusion_matrix(model: Model, x_test, y_test, labels, marker):
    predictions = np.argmax(model.predict(x_test), axis=1)
    y_test_flat = np.argmax(y_test, axis=1)
    cm = confusion_matrix(y_test_flat, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    _, ax = plt.subplots(figsize=(10, 10))
    disp.plot(xticks_rotation='vertical', ax=ax, cmap='summer')
    timestamp = datetime.now().strftime("%d%m%Y")
    plt.savefig(f'confusion_matrix_{marker}_{timestamp}.png')
    plt.show()


def show_misclassified(model: Model, x_test, y_test, labels, marker, n=5):
    predictions = np.argmax(model.predict(x_test), axis=1)
    y_test_flat = np.argmax(y_test, axis=1)
    incorrect_indices = np.nonzero(predictions != y_test_flat)[0]
    for i in range(n):
        idx = incorrect_indices[i]
        plt.imshow(x_test[idx])
        plt.xlabel(f"True: {labels[y_test_flat[idx]]}, Predicted: {labels[predictions[idx]]}")
        plt.title(f"{marker} - Misclassified #{i + 1}")
        plt.show()
