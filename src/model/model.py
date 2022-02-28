import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict
from tqdm import tqdm



sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.model.convolution import Convolution
from src.model.pooling import Pooling
from src.model.som import SOM
from src.model.activation import Relu, Tanh
from data.mnist import mnist

class SelfOrganizingConvNet:

    def __init__(self, input_dim=(1,28,28), \
                conv_param={'filter_num':6, 'filter_size':2, 'pad':0, 'stride':1}, \
                som_param={'map_size':50, 'input_vec':[1,12,24,24], 'alpha':1.0},  weight_init_std=0.01) -> None:
        filter_num = conv_param['filter_num'] # フィルター数
        filter_size = conv_param['filter_size'] # フィルタの縦横
        filter_pad = conv_param['pad'] # パディング
        filter_stride = conv_param['stride'] # ストライド
        input_size = input_dim[1] # 入力データの縦横
        conv_output_size = (input_size - filter_size + 2 * filter_pad) / filter_stride + 1 # Convレイヤの出力の縦横
        pool_output_size = int(filter_num * (conv_output_size / 2) * (conv_output_size / 2)) # Poolレイヤの出力の縦横

        # パラメータの初期値を設定
        self.params = {} # 初期化
        self.params['W1'] = weight_init_std * np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        self.params['b1'] = np.zeros(filter_num)
        self.params['W2'] = weight_init_std * np.random.randn(12, 6, filter_size, filter_size)
        self.params['b2'] = np.zeros(12)

        ## レイヤを格納したディクショナリ変数を作成
        self.layers = OrderedDict() # 順番付きディクショナリ変数を初期化

        # Conv_Pooling層
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'], \
                                           conv_param['stride'], conv_param['pad'])
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2)

        self.layers['Conv2'] = Convolution(self.params['W2'], self.params['b2'], \
                                           conv_param['stride'], conv_param['pad'])
        self.layers['Relu2'] = Relu()
        self.layers['Pool2'] = Pooling(pool_h=2, pool_w=2)

        # Self_organizing層
        self.layers['SOM1'] = SOM(som_param['map_size'], som_param['alpha'], 2, som_param['input_vec'])
        self.layers['Tanh1'] = Tanh()




    def self_organizing(self, x):
        # 自己組織化

        x = self.layers['Conv1'].forward(x)
        x = self.layers['Relu1'].forward(x)
        x = self.layers['Pool1'].forward(x)
        x = self.layers['Conv2'].forward(x)
        x = self.layers['Relu2'].forward(x)
        x = self.layers['Pool2'].forward(x)
        self.layers['SOM1'].self_organization(x)
        x = self.layers['SOM1'].forward(x)
        x = self.layers['Tanh1'].forward(x)

        return x

if __name__ == '__main__':
    model = SelfOrganizingConvNet()
    mnist_data = mnist()
    mnist_data.Read_data(0)
    sample_data = []
    for i in range(10):
        sample_data.append(mnist_data.data_vec[i,5])
        sample_data[i]=sample_data[i].reshape(1,1,28,28)


    # plt.figure()
    # sns.heatmap(sample_data_0.reshape(28,28))
    # plt.figure()
    # sns.heatmap(sample_data_1.reshape(28,28))
    # plt.figure()
    # sns.heatmap(sample_data_9.reshape(28,28))

    # plt.show()



    for class_num in range(10):
        print(class_num)
        for _ in tqdm(range(500)):
            # class_num =  np.random.randint(0,10)
            num =  np.random.randint(0,100)
            data = mnist_data.data_vec[class_num,num]
            data = data.reshape(1,1,28,28)
            model.self_organizing(data)

    # for layer in model.layers.values():
    #     for i in range(10):
    #         sample_data[i] = layer.forward(sample_data[i])

    for i in range(10):
        plt.figure()
        weight = model.layers['SOM1'].weight[0,0]
        weight = weight.reshape(12,24,24)
        sns.heatmap(weight[i])
    plt.figure()
    sns.heatmap(sample_data[i])
    plt.show()
