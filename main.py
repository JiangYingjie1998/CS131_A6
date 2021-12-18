import pandas as pd
import matplotlib.pyplot
import numpy as np
import random
from sklearn.model_selection import train_test_split


# class Iris(object):
#     def __init__(self, list):
#         self.__init(float(list[0]), float(list[1]), float(list[2]), float(list[3]), list[4])
#
#     def __init(self, sepal_length, sepal_width, petal_length, petal_width, type):
#         self.sepal_length = sepal_length
#         self.sepal_width = sepal_width
#         self.petal_length = petal_length
#         self.petal_width = petal_width
#         self.type = type


# def load_data(fileName):
#     """
#     read from file
#     :return list, each element is like []
#     """
#     data = []
#     with open(fileName, "r") as f:
#         for line in f.readlines():
#             splits = line.strip().split(",")
#             data.append(Iris(splits))
#     print("---data size:", len(data), "--------")
#     return data


def init_param(n_x, n_h, n_y):
    np.random.seed(131)
    w1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros(shape=(n_h, 1))
    w2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros(shape=(n_y, 1))

    return w1, b1, w2, b2


def sigmoid(n):
    return 1 / (1 + np.exp(-n))


def forward(X, param):
    w1 = param[0]
    b1 = param[1]
    w2 = param[2]
    b2 = param[3]

    z1 = np.dot(w1, X) + b1
    a1 = np.tanh(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = sigmoid(z2)

    temp = (z1, a1, z2, a2)

    return a2, temp


def cost_fun(a2, Y):
    m = Y.shape[1]
    # f(x) = ln(x)*Y + (1-Y)*ln(1-x)
    logs = np.multiply(np.log(a2), Y) + np.multiply((1 - Y), np.log(1 - a2))
    cost = - np.sum(logs) / m
    return cost


def cal_grads(param, temp, X, Y):
    m = Y.shape[1]

    w2 = param[2]
    a1 = temp[1]
    a2 = temp[3]

    dz2 = a2 - Y
    dw2 = (1 / m) * np.dot(dz2, a1.T)
    db2 = (1 / m) * np.sum(dz2, axis=1, keepdims=True)
    dz1 = np.multiply(np.dot(w2.T, dz2), 1 - np.power(a1, 2))
    dw1 = (1 / m) * np.dot(dz1, X.T)
    db1 = (1 / m) * np.sum(dz1, axis=1, keepdims=True)

    return dw1, db1, dw2, db2


def backward(param, grads, nita=0.2):
    w1 = param[0]
    b1 = param[1]
    w2 = param[2]
    b2 = param[3]

    dw1 = grads[0]
    db1 = grads[1]
    dw2 = grads[2]
    db2 = grads[3]

    w1 -= nita * dw1
    b1 -= nita * dw1
    w2 -= nita * dw1
    b2 -= nita * dw1

    return w1, b1, w2, b2


def init_network(X, Y, n_h, n_x, n_y, loop_times=5000):
    np.random.seed(131)

    param = init_param(n_x, n_h, n_y)

    for i in range(loop_times):
        result, temp = forward(X, param)
        cost = cost_fun(result, Y)

        backward(param, cal_grads(param, temp, X, Y))

    return param


def predict(param, x_test, y_test):
    w1 = param[0]
    b1 = param[1]
    w2 = param[2]
    b2 = param[3]

    z1 = np.dot(w1, x_test) + b1
    a1 = np.tanh(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = sigmoid(z2)

    output = np.empty(shape=(y_test.shape[0], y_test.shape[1]), dtype=int)
    for i in range(y_test.shape[0]):
        for j in range(y_test.shape[1]):
            if a2[i][j] > 0.5:
                output[i][j] = 1
            else:
                output[i][j] = 0

    correct_cnt = 0
    for i in range(y_test.shape[1]):
        if output[0][i] == y_test[0][i] \
                and output[1][i] == y_test[1][i] \
                and output[2][i] == y_test[2][i]:
            correct_cnt += 1
    acc_rate = correct_cnt / int(y_test.shape[1]) * 100
    print("accuracy rate: {.2f}".format(acc_rate))

    return output


if __name__ == '__main__':
    types = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
    attributes = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
    fileName = "ANN - Iris data.txt"
    iris = pd.read_csv(fileName, names=attributes)

    # X = iris.iloc[:, 0:4].values.T
    # Y = iris.iloc[:,5]
    # x_train, y_train, x_test, y_test = train_test_split(iris)
    train,test = train_test_split(dat)
