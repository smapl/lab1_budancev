import numpy as np

from matplotlib import pyplot as plt
from loguru import logger

# функция модели
def compute_hypothesis(X, theta):
    return np.matmul(X, theta)


# функция стоимости
def compute_cost(X, y, theta):
    m = X.shape[0]

    return 1 / (2 * m) * sum((compute_hypothesis(X, theta) - y) ** 2)


# градиентный спуск
def gradient_descend(X, y, theta, alpha, num_iter):

    history = list()

    m = X.shape[0]

    for _ in range(num_iter):

        theta_temp = theta
        theta_temp[0] = (
            theta_temp[0] - alpha * (compute_hypothesis(X, theta) - y).dot(X[:, 0]) / m
        )
        theta_temp[1] = (
            theta_temp[1] - alpha * (compute_hypothesis(X, theta) - y).dot(X[:, 1]) / m
        )
        theta = theta_temp

        history.append(compute_cost(X, y, theta))

    return history, theta


def load_data(data_file_path):

    with open(data_file_path) as input_file:
        X = list()
        y = list()

        for line in input_file:

            *row, label = map(float, line.split(","))
            X.append([1] + row)
            y.append(label)

        return np.array(X, float), np.array(y, float)


if __name__ == "__main__":

    X, y = load_data(
        "/home/smapl/univer/budancev_labs/lab_1/lab1_budancev/src/lab1data1.txt"
    )

    plt.title("Исходные данные")
    plt.scatter(X[:, 1], y, c="r", marker="x")
    plt.show()

    logger.info(
        "Значение функции стоимости при theta = [0, 0]: ",
        compute_cost(X, y, np.array([0, 0])),
    )

    history, theta = gradient_descend(X, y, np.array([0, 0], float), 0.01, 1500)

    plt.title("График изменения функции стоимости от номера итерации")
    plt.plot(range(len(history)), history)
    plt.show()

    plt.title("Обученая модель")
    plt.scatter(X[:, 1], y, c="r", marker="x")
    plt.plot(X[:, 1], compute_hypothesis(X, theta))
    plt.show()
