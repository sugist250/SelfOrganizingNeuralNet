
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.model.activation import Relu, Tanh
from collections import OrderedDict

# np.random.seed(seed=56)

class SOM:
    def __init__(self, map_size, alpha, radius, input_vec) -> None:
        self.map_size = map_size
        self.alpha = alpha
        self.radius = radius
        self.input_vec_size = np.prod(input_vec)
        self.max_dist = np.linalg.norm(np.array((0,0))-np.array((map_size-1,map_size-1)))

        self.weight = np.random.uniform(low=-1.0, high=1.0,size=(map_size, map_size, self.input_vec_size))

    def forward(self, x):
        out = []
        for i in range(self.map_size):
            for j in range(self.map_size):
                out.append(np.dot(self.weight[i,j], x.flatten()))
        out = np.array(out)
        return out.reshape(self.map_size,self.map_size)

    def self_organization(self, x):

        min_idx = self.__return_min_index2(x)
        self.__update_weight2(x, min_idx)

    def __update_weight(self, x, min_idx):
        min_idx_x = int(min_idx / (self.map_size-(2*self.radius))) + self.radius
        min_idx_y = int(min_idx % (self.map_size-(2*self.radius))) + self.radius
        for i in range(-self.radius, self.radius):
            for j in range(-self.radius, self.radius):
                update_value = self.__return_update_value(x, min_idx_x, min_idx_y, i, j)
                self.weight[min_idx_x+i,min_idx_y+j] += update_value

    def __update_weight2(self, x, min_idx):
        min_idx_x = int(min_idx / (self.map_size))
        min_idx_y = int(min_idx % (self.map_size))
        for i in range(self.map_size):
            for j in range(self.map_size):
                update_value = self.__return_update_value(x, min_idx_x, min_idx_y, i, j)
                self.weight[i,j] += update_value


    def __return_min_index(self, x):
        resize_min = self.radius
        resize_max = self.map_size - self.radius
        resize_map = self.weight[resize_min:resize_max, resize_min:resize_max]
        min_idx = np.argmin(((x.flatten()-resize_map)**2).sum(axis=2))
        return min_idx

    def __return_min_index2(self, x):
        min_idx = np.argmin(((x.flatten()-self.weight)**2).sum(axis=2))
        return min_idx

    def __return_update_value(self, x, min_idx_x, min_idx_y, i, j):
        # self.alpha = float(self.alpha - (1/1000))
        diff = x.flatten() - self.weight[i,j]
        # update_value = self.alpha * diff  * self.__neighborhood_function(i,j)
        update_value = self.alpha * diff  * self.__neighborhood_function2(min_idx_x, min_idx_y, i,j)
        return update_value

    def __neighborhood_function(self, i,j):
        x = 1 - (abs(i)/(self.radius + 1))
        y = 1 - (abs(j)/(self.radius + 1))
        return (x+y)/2

    def __neighborhood_function2(self, min_idx_x, min_idx_y,i,j):
        a = np.array([min_idx_x, min_idx_y])
        b = np.array([i,j])
        dist = np.linalg.norm(a-b)
        return 1 - (dist/self.max_dist)

if __name__ == '__main__':

    som_size = 30
    alpha = 0.08
    radius = 3

    inputs_vec_size = 10
    w = np.random.uniform(low=-1.0, high=1.0,size=(som_size, som_size, inputs_vec_size))
    input_1 = np.random.uniform(low=-1.0, high=1.0,size=(inputs_vec_size))
    input_2 = np.random.uniform(low=-1.0, high=1.0,size=(inputs_vec_size))
    input_3 = np.random.uniform(low=-1.0, high=1.0,size=(inputs_vec_size))

    layers = OrderedDict()
    layers['som'] = SOM(som_size, alpha, radius, inputs_vec_size)
    layers['tanh'] = Tanh()

    for _ in tqdm(range(100)):
        layers['som'].self_organization(input_1)
        layers['som'].self_organization(input_2)
        layers['som'].self_organization(input_3)

    x1 = input_1
    x2 = input_2
    x3 = input_3
    for layer in layers.values():
        x1 = layer.forward(x1)
        x2 = layer.forward(x2)
        x3 = layer.forward(x3)

    print(input_1)
    print(input_2)
    print(input_3)
    plt.figure()
    sns.heatmap(x1)
    plt.figure()
    sns.heatmap(x2)
    plt.figure()
    sns.heatmap(x3)
    plt.show()







