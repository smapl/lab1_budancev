import numpy as np

from matplotlib import pyplot as plt
from loguru import logger
import time


# функция моедели
def compute_hypothesis(X, theta):
    return X @ theta


# функция стоимости
# пишет nan из-за переполнения в квадрате (того что возовдится в квадрат)
# так же засчет этого появляется предупреждение RuntimeWarning
def compute_cost(X, y, theta):
    m = X.shape[0]
    try:
        res = 1 / (2 * m) * sum((compute_hypothesis(X, theta) - y) ** 2)

    except RuntimeWarning:
        logger.error(
            f"in square {compute_hypothesis(X, theta) - y}\nX={X}\ntheta={theta}\ny={y}"
        )

    logger.info(res)
    return res


# градиентный спуск
def gradient_descend(X, y, theta, alpha, num_iter):
    # theta = np.array([0, 0, 0]
    # alpha = 0.01
    # num_iter = 1500
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


# стандартизация признаков
def scale_features(X):

    new_X = X[:, 1:]
    # np.mean - срденее арифметическое значение массива
    # np.std - средне квадратичное отклонениезначений элементов массива
    z = (new_X - np.mean(new_X, axis=0)) // np.std(new_X, axis=0)
    array_one = np.ones((z.shape[0], 1))  # массив заполнен единицами

    finish_z = np.column_stack((array_one, z))  # соединяет массивы повертикали

    logger.info(z)
    logger.info(array_one)
    logger.info(finish_z)

    return finish_z


# нормальное уравнение
# выводит точную theta, т.к. градиентный спуск дает лишь приближенную
# а иногда градиентный спуск может и вовсе обосраться
# ошибка операнды не могут транслироваться вместе с фигурами
# Exception has occurred: ValueError


def normal_equation(X, y):
    # функция вычисляет псевдообратную матрицу
    theta = np.linalg.pinv((X.T @ X)) @ (X.T @ y)  # @ == умножить в np array

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

    X, y = load_data(
        "/home/smapl/univer/budancev_labs/lab_1/lab1_budancev/src/lab1data2.txt"
    )

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
