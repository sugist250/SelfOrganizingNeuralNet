import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from data.mnist import mnist
from src.model.activation import Tanh
np.random.seed(seed=56)


class SOM:
    def __init__(self, Nx, Ny, input_vec, alpha, radius) -> None:
        self.N_x = Nx
        self.N_y = Ny
        self.input_vec = input_vec
        self.alpha = alpha
        self.radius = radius
        # self.weight = np.zeros([Nx, Ny, input_vec[0], input_vec[1], input_vec[2], input_vec[3]])
        self.weight = np.random.uniform(low=0.0, high=1.0, size=(Nx, Ny, input_vec[0]*input_vec[1]*input_vec[2]*input_vec[3]))

    def update_weight(self, inputs):
        inputs_vec = inputs.flatten()
        min_index = np.argmin(((self.weight - inputs_vec)**2).sum(axis=(2)))
        mini = int(min_index / self.N_y)
        minj = int(min_index % self.N_y)
        for i in range(-self.radius, self.radius+1):
            for j in range(-self.radius, self.radius+1):
                try:
                    noize = np.random.uniform(low=-1.0, high=1.0, size=(self.input_vec[0]*self.input_vec[1]*self.input_vec[2]*self.input_vec[3]))
                    self.weight[mini + i, minj + j] += self.alpha * (inputs_vec - self.weight[mini + i, minj + j])
                except:
                    pass


    def forward(self, inputs):
        inputs_vec = inputs.flatten()
        out = np.zeros((self.N_x, self.N_y))
        for i in range(self.N_x):
            for j in range(self.N_y):
                out[i, j] = self.return_similarity(inputs_vec, self.weight[i, j])
        return out


    def return_similarity(self, vec_1, vec_2, mode='euclid'):
        # 二つのベクトルの類似度を計算する
        # デフォルトはユークリッド距離
        if mode == 'euclid':
            return 0.1 / np.linalg.norm(vec_1 - vec_2)
        elif mode == 'cosine':
            return np.dot(vec_1.flatten(), vec_2.flatten()) / (np.linalg.norm(vec_1.flatten()) * np.linalg.norm(vec_2.flatten()))


if __name__ == '__main__':
    vec = [1,1,14,14]
    som = SOM(30,30,vec,0.08,2)
    t = Tanh()

    mnist_data = mnist()
    mnist_data.Read_data(0)
    sample_data_0 = mnist_data.data_vec[0,1]
    sample_data_0 = sample_data_0.reshape(1,1,14,14)
    for _ in range(10000):
        print('update_weight')
        som.update_weight(sample_data_0)
    x = som.forward(sample_data_0)
    # print(x)
    # print('')
    x = t.forward(x)


    print(x)


