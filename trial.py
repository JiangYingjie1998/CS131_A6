# -*- coding: utf-8 -*-
import math
import random
import sys
import os
import json

import numpy as np
import pandas as pd

flowerLables = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}
random.seed(131)


def rand(a, b):
    return (b - a) * random.random() + a


# build a I*J matrix
def makeMatrix(I, J, fill=0.0):
    m = []
    for i in range(I):
        m.append([fill] * J)
    return m


# 函数 sigmoid，这里采用 tanh，因为看起来要比标准的 1/(1+e^-x) 漂亮些
def sigmoid(x):
    return np.tanh(x)


# 函数 sigmoid 的派生函数, 为了得到输出 (即：y)
def dsigmoid(y):
    return 1.0 - y ** 2


class NN:
    """
    3-layer BP
    """

    def __init__(self, ni, nh, no):
        # number of input,hidden and output layers
        self.ni = ni + 1
        self.nh = nh
        self.no = no

        self.ai = [1.0] * self.ni
        self.ah = [1.0] * self.nh
        self.ao = [1.0] * self.no

        self.wi = makeMatrix(self.ni, self.nh)
        self.wo = makeMatrix(self.nh, self.no)

        for i in range(self.ni):
            for j in range(self.nh):
                self.wi[i][j] = rand(-0.2, 0.2)
        for j in range(self.nh):
            for k in range(self.no):
                self.wo[j][k] = rand(-2.0, 2.0)

        self.ci = makeMatrix(self.ni, self.nh)
        self.co = makeMatrix(self.nh, self.no)

    def update(self, inputs):
        if len(inputs) != self.ni - 1:
            raise

        for i in range(self.ni - 1):
            self.ai[i] = inputs[i]

        for j in range(self.nh):
            sum = 0.0
            for i in range(self.ni):
                sum = sum + self.ai[i] * self.wi[i][j]
            self.ah[j] = sigmoid(sum)

        for k in range(self.no):
            sum = 0.0
            for j in range(self.nh):
                sum = sum + self.ah[j] * self.wo[j][k]
            self.ao[k] = sigmoid(sum)

        return self.ao[:]

    def backward(self, targets, N, M):
        output_delta = [0.0] * self.no
        for k in range(self.no):
            error = targets[k] - self.ao[k]
            output_delta[k] = dsigmoid(self.ao[k]) * error

        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error = error + output_delta[k] * self.wo[j][k]
            hidden_deltas[j] = dsigmoid(self.ah[j]) * error

        for j in range(self.nh):
            for k in range(self.no):
                change = output_delta[k] * self.ah[j]
                self.wo[j][k] = self.wo[j][k] + N * change + M * self.co[j][k]
                self.co[j][k] = change
                # print(N * change, M * self.co[j][k])

        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j] * self.ai[i]
                self.wi[i][j] = self.wi[i][j] + N * change + M * self.ci[i][j]
                self.ci[i][j] = change

        # 计算误差
        error = 0.0
        for k in range(len(targets)):
            error = error + 0.5 * (targets[k] - self.ao[k]) ** 2
            # error += 0.5 * (targets[k] - self.ao[k]) ** 2
        return error

    def test(self, patterns):
        count = 0
        for p in patterns:
            target = flowerLables[(p[1].index(1))]
            result = self.update(p[0])
            index = result.index(max(result))
            # print(p[0], ':', target, '->', flowerLables[index])
            count += (target == flowerLables[index])
        accuracy = float(count / len(patterns))
        print('accuracy: %-.9f' % accuracy)

    def weights(self):
        for i in range(self.ni):
            print(self.wi[i])
        for j in range(self.nh):
            print(self.wo[j])

    def train(self, patterns, iterations=1000, N=0.1, M=0.01):
        """
        N: learning rate
        M: momentum factor
        """
        for i in range(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                error = error + self.backward(targets, N, M)
            if i % 100 == 0:
                print('error rate: %-.9f' % error)


types = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
attributes = ["sepalLength", "sepalWidth", "petalLength", "petalWidth", "species"]
fileName = "ANN - Iris data.txt"


def iris():
    # read dataset
    raw = pd.read_csv(fileName, names=attributes)
    raw_data = raw.values
    raw_feature = raw_data[0:, 0:4]

    data = []
    for i in range(len(raw_feature)):
        ele = []
        ele.append(list(raw_feature[i]))
        if raw_data[i][4] == types[0]:
            ele.append([1, 0, 0])
        elif raw_data[i][4] == types[1]:
            ele.append([0, 1, 0])
        else:
            ele.append([0, 0, 1])
        data.append(ele)

    # print (data)
    random.shuffle(data)
    # print (data)

    training = data[0:100]
    test = data[101:]
    nn = NN(4, 7, 3)
    nn.train(training, iterations=1000)

    # save weights
    with open('wi.txt', 'w+') as wif:
        json.dump(nn.wi, wif)
    with open('wo.txt', 'w+') as wof:
        json.dump(nn.wo, wof)

    nn.test(test)


def query():
    if not os.path.exists("wi.txt"):
        print("model has to be trained before query")
    if not os.path.exists("wo.txt"):
        print("model has to be trained before query")
    nn = NN(4, 7, 3)
    with open('wi.txt', 'r') as wif:
        nn.wi = json.load(wif)
    with open('wo.txt', 'r') as wof:
        nn.wo = json.load(wof)

    inputs = []
    for i in attributes[:-1]:
        inputs = float(input("Please input a float as " + i))
    nn.test([inputs])


if __name__ == '__main__':
    while True:
        option = input("1.train\n2.query\n3.exit\n")
        # option = "1"
        if option == "1":
            iris()
        elif option == "2":
            query()
        else:
            sys.exit(0)
