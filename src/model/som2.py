
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.model.activation import Relu, Tanh
from collections import OrderedDict

# np.random.seed(seed=56)

class SOM:
    def __init__(self, map_size, alpha, radius, input_vec_size) -> None:
        self.map_size = map_size
        self.alpha = alpha
        self.radius = radius

        self.weight = np.random.uniform(low=-1.0, high=1.0,size=(map_size, map_size, input_vec_size))

    def forward(self, x):
        out = []
        for i in range(self.map_size):
            for j in range(self.map_size):
                out.append(np.dot(self.weight[i,j], x))
        out = np.array(out)
        return out.reshape(self.map_size,self.map_size)

    def self_organization(self, x):
        min_idx = self.__return_min_index(x)
        self.__update_weight(x, min_idx)

    def __update_weight(self, x, min_idx):
        min_idx_x = int(min_idx / self.map_size)
        min_idx_y = int(min_idx % self.map_size)
        for i in range(-self.radius, self.radius):
            for j in range(-self.radius, self.radius):
                update_value = self.__return_update_value(x, min_idx, i, j)
                self.weight[min_idx_x+i,min_idx_y+j] += update_value

    def __return_min_index(self, x):
        resize_min = self.radius
        resize_max = self.map_size - self.radius
        return np.argmin(((self.weight[resize_min:resize_max, resize_min:resize_max]-x)**2).sum(axis=2))

    def __return_update_value(self, x, min_idx, i, j):
        min_idx_x = int(min_idx / self.map_size) + self.radius
        min_idx_y = int(min_idx % self.map_size) + self.radius
        diff = x - self.weight[min_idx_x+i,min_idx_y+j]
        update_value = self.alpha * diff #  * self.__neighborhood_function(i,j)
        return update_value

    def __neighborhood_function(self, i,j):
        x = 1 - (abs(i)/(self.radius + 1))
        y = 1 - (abs(j)/(self.radius + 1))
        return (x+y)/2

if __name__ == '__main__':

    som_size = 10
    alpha = 0.5
    radius = 3

    inputs_vec_size = 1
    w = np.random.uniform(low=-1.0, high=1.0,size=(som_size, som_size, inputs_vec_size))
    input_1 = np.random.uniform(low=-1.0, high=1.0,size=(inputs_vec_size))
    input_2 = np.random.uniform(low=-1.0, high=1.0,size=(inputs_vec_size))
    input_3 = np.random.uniform(low=-1.0, high=1.0,size=(inputs_vec_size))

    for _ in range(100000):
        #print(np.argmin(((w[radius:som_size-(radius), radius:som_size-(radius)]-input_1)**2).sum(axis=2)))
        a = w[radius:som_size-(radius), radius:som_size-(radius)]
        s = np.argmin(((a-input_1)**2).sum(axis=2))

        if int(s/som_size)+radius >= som_size or int(s%som_size)+radius >= som_size:
            print(s)
            print(int(s/som_size)+radius)
            print(int(s%som_size)+radius)
    # layers = OrderedDict()
    # layers['som'] = SOM(som_size, alpha, radius, inputs_vec_size)
    # layers['tanh'] = Tanh()

    # for _ in range(1000):
    #     layers['som'].self_organization(input_1)
    #     layers['som'].self_organization(input_2)
    #     layers['som'].self_organization(input_3)

    # x1 = input_1
    # x2 = input_2
    # x3 = input_3
    # for layer in layers.values():
    #     x1 = layer.forward(x1)
    #     x2 = layer.forward(x2)
    #     x3 = layer.forward(x3)


    # plt.figure()
    # sns.heatmap(x1)
    # plt.figure()
    # sns.heatmap(x2)
    # plt.figure()
    # sns.heatmap(x3)
    # plt.show()







