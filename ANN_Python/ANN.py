import numpy as np
import random as rand
import math
from timeit import default_timer as timer


class ANN:
    def __init__(self, num_layers=3, bias=1, activation=1, layers=[]):
        self.num_layers = num_layers
        self.layers = layers
        self.weights = []
        self.delta_weights = []
        self.neurons = []
        self.sums = []
        self.bias = bias
        self.activation = activation

        for i in range(self.num_layers):
            self.weights.append([])
            if i > 0:
                num_weights = self.layers[i - 1] * self.layers[i]
                if self.bias:
                    num_weights = num_weights + self.layers[i]

                for j in range(num_weights):
                    self.weights[i - 1].append(rand.gauss(0, 5))

    def sigmoid(self, x):
        if x > 100:
            return 0
        if x < -100:
            return 1

        return 1 / (1 + math.exp(-x))

    def activation_function(self, val):
        if self.activation == 0:
            return self.sigmoid(val)
        if self.activation == 1:
            return np.tanh(val)
        if self.activation > 1:
            return self.sigmoid(val)

    def run(self, inputs=[]):
        self.neurons = []
        self.neurons.append([])
        self.sums = []
        self.sums.append([])
        self.neurons[0] = inputs

        for i in range(1, self.num_layers):  # go through all layers
            self.neurons.append([])
            self.sums.append([])
            for j in range(self.layers[i]):  # current layer neurons
                weighted_sum = 0
                for k in range(self.layers[i - 1]):  # previous layer neurons
                    weighted_sum = weighted_sum + \
                                   self.neurons[i - 1][k] * self.weights[i - 1][j * self.layers[i - 1] + k]
                if self.bias:
                    weighted_sum = weighted_sum + self.weights[i - 1][self.layers[i] * self.layers[i - 1] + j]
                self.sums[i].append(weighted_sum)
                self.neurons[i].append(self.activation_function(weighted_sum))

        return self.neurons[self.num_layers - 1]

# ann = ANN(3, True, 1, [20, 20, 4])
#
# start = timer()
#
# in_arr = [1 for x in range(20)]
# out = ann.run(in_arr)
#
# end = timer()
#
# print "Time taken:", (end - start)







