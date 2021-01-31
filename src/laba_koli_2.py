import numpy as np
from matplotlib import pyplot as plt


def compute_hypothesis(X, theta):
    return X @ theta


def compute_cost(X, y, theta):
    m = X.shape[0]  # количество примеров в выборке
    return 1 / (2 * m) * sum((compute_hypothesis(X, theta) - y) ** 2)


def gradient_descend(X, y, theta, alpha, num_iter):
    history = list()
    m = X.shape[0]  # количество примеров в выборке
    n = X.shape[1]  # количество признаков с фиктивным

    for i in range(num_iter):
        theta_temp = theta
        theta_temp = theta_temp - alpha * (X.T @ (compute_hypothesis(X, theta) - y)) / m
        theta = theta_temp
        history.append(compute_cost(X, y, theta))

    return history, theta


def scale_features(X):
    # ВАШ КОД ЗДЕСЬ
    for i in range(1, (X.shape[1])):
        x_mid = np.mean(X[:, i])
        x_sigma = np.std(X[:, i])
        X[:, i] = (X[:, i] - x_mid) / x_sigma
    return X
    # pass
    # =====================


def normal_equation(X, y):
    # ВАШ КОД ЗДЕСь
    return (np.linalg.inv(X.T @ X)) @ (X.T @ y)
    # pass
    # =====================


def load_data(data_file_path):
    with open(data_file_path) as input_file:
        X = list()
        y = list()
        for line in input_file:
            *row, label = map(float, line.split(","))
            X.append([1] + row)
            y.append(label)
        return np.array(X, float), np.array(y, float)


X, y = load_data("lab1data2.txt")

history, theta = gradient_descend(X, y, np.array([0, 0, 0], float), 0.01, 1500)

plt.title("График изменения функции стоимости от номера итерации до нормализации")
plt.plot(range(len(history)), history)
plt.show()

X = scale_features(X)

history, theta = gradient_descend(X, y, np.array([0, 0, 0], float), 0.4, 1500)

plt.title("График изменения функции стоимости от номера итерации после нормализации")
plt.plot(range(len(history)), history)
plt.show()

theta_solution = normal_equation(X, y)
print(
    f"theta, посчитанные через градиентный спуск: {theta}, через нормальное уравнение: {theta_solution}"
)
