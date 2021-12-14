import pandas as pd
import matplotlib.pyplot
import numpy as np

types = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]


class Iris(object):

    def __init__(self, sepal_length, sepal_width, petal_lenth, petal_width, type):
        self.sepal_length = sepal_length
        self.sepal_width = sepal_width
        self.petal_lenth = petal_lenth
        self.petal_width = petal_width
        self.type = type


def read_file():
    """
    read from file
    :return list, each element is like []
    """
    data = []
    with open("ANN - Iris data.txt", "r") as f:
        for line in f.readlines():
            splits = line.strip().split(",")
            data.append(splits)

    i = 0
    while i < 3:
        data.append(lines[i:i + 50])
        for d in data:
            print("\t".join(d))
    print("sepal_length|\tsepal_width|\tpetal_lenth|\tpetal_width")


if __name__ == '__main__':
    read_file()
