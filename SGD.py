import numpy as np
import random
from my_helpers import compute_loss
from my_helpers import calculate_accuracy
from my_helpers import load_csv_data

def calculate_stochastic_gradient(y, X, w, lambda_, n, num_examples):

    def is_support(y_n, x_n, w):
        return y_n * x_n @ w < 1

    x_n, y_n = X[n], y[n]

    grad = - y_n * x_n.T if is_support(y_n, x_n, w) else np.zeros_like(x_n.T)
    grad = num_examples * np.squeeze(grad) + lambda_ * w
    return grad


def main():

    # Downloaded from https://inclass.kaggle.com/c/epfml-project-1/data
    # Just to test if SGD works
    DATA_TRAIN_PATH = 'train.csv'

    # load_csv_data is for this data only. Ask the teacher how he will give the data to parse it in a good way.
    y, X, ids = load_csv_data(DATA_TRAIN_PATH, sub_sample=True)
    print(y.shape, X.shape)

    # Master will give X, y, max_iter, gamma, lambda
    #X =
    #y =
    max_iter = 100000
    gamma = 1
    lambda_ = 0.01

    num_examples, num_features = X.shape
    w = np.zeros(num_features)

    #Begin iteration for SGD
    for it in range(max_iter):
        n = random.randint(0, num_examples - 1)

        grad = calculate_stochastic_gradient(y, X, w, lambda_, n, num_examples)
        w -= gamma / (it + 1) * grad

        # Now share w with other slaves and average result and continue

        if(it%10000 == 0):
            print(it)
            #loss = compute_loss(y, X, w)
            #print(loss)

    accuracy = calculate_accuracy(y, X, w)
    print("Accuracy = ", accuracy)


if __name__ == '__main__':
    main()