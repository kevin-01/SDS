import numpy as np
import csv

def compute_loss(y, tx, w):
    e = y - tx @ w
    mse = np.sum(e*e)/(2*len(e))
    return mse

def accuracy(y1, y2):
    return np.mean(y1 == y2)


def prediction(X, w):
    return (X @ w > 0) * 2 - 1


def calculate_accuracy(y, X, w):
    predicted_y = prediction(X, w)
    return accuracy(predicted_y, y)

def load_csv_data(data_path, sub_sample=False):
    """Loads data.
    return
        y(class labels), tX (features) and ids (event ids).
    """
    y = np.genfromtxt(
        data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(
        data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y == 'b')] = -1

    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids

def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix."""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1

    return y_pred