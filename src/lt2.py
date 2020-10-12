import numpy as np

from matplotlib import pyplot as plt
from loguru import logger


def compute_hypothesis(X, theta):
    return X @ theta


def compute_cost(X, y, theta):
    m = X.shape[0]

    logger.info(1 / (2 * m) * sum((compute_hypothesis(X, theta) - y) ** 2))
    return 1 / (2 * m) * sum((compute_hypothesis(X, theta) - y) ** 2)


def gradient_descend(X, y, theta, alpha, num_iter):
    history = list()
    m = X.shape[0]  # количество примеров в выборке
    n = X.shape[1]  # количество признаков с фиктивным
    for _ in range(num_iter):
        for j in range(n):
            theta_temp = theta
            theta_temp[j] = (
                theta_temp[j]
                - alpha * (compute_hypothesis(X, theta) - y).dot(X[:, j]) / m
            )
            theta = theta_temp

            history.append(compute_cost(X, y, theta))
    logger.info(history)
    logger.info(theta)
    return history, theta


def scale_features(X):

    new_X = X[:, 1:]
    z = (new_X - np.mean(new_X, axis=0)) // np.std(new_X, axis=0)
    array_one = np.ones((z.shape[0], 1))

    finish_z = np.column_stack((array_one, z))

    logger.info(z)
    logger.info(array_one)
    logger.info(finish_z)

    return finish_z


def normal_equation(X, y):
    theta = np.linalg.pinv(np.transpose(X) * y) * np.transpose(X) * y

    logger.info(theta)
    return theta


def load_data(data_file_path):
    with open(data_file_path) as input_file:

        X, y = [], []
        for line in input_file:
            *row, label = map(float, line.split(","))
            X.append([1] + row)
            y.append(label)

        return np.array(X, float), np.array(y, float)


if __name__ == "__main__":

    X, y = load_data("lab1data2.txt")

    history, theta = gradient_descend(X, y, np.array([0, 0, 0], float), 0.01, 1500)

    plt.title("График изменения функции стоимости от номера итерации до нормализации")
    plt.plot(range(len(history)), history)
    plt.show()

    X = scale_features(X)

    history, theta = gradient_descend(X, y, np.array([0, 0, 0], float), 0.01, 1500)

    plt.title(
        "График изменения функции стоимости от номера итерации после нормализации"
    )
    plt.plot(range(len(history)), history)
    plt.show()

    theta_solution = normal_equation(X, y)
    logger.info(
        f"theta, посчитанные через градиентный спуск: {theta}, через нормальное уравнение: {theta_solution}"
    )
