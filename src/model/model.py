import sys
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict
from tqdm import tqdm



sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from model.layer.convolution import Convolution
from model.layer.pooling import Pooling
from model.layer.som import SOM
from model.layer.activation import Relu, Tanh
from data.mnist import mnist

class SelfOrganizingConvNet:

    def __init__(self, input_dim=(1,28,28), \
                conv_param={'filter_num':12, 'filter_size':3, 'pad':0, 'stride':1}, \
                som_param={'map_size':40, 'input_vec':[1,12,22,22], 'alpha':1.0}) -> None:
        filter_num = conv_param['filter_num'] # フィルター数
        filter_size = conv_param['filter_size'] # フィルタの縦横
        self.som_input_size = som_param['input_vec']

        ## レイヤを格納したディクショナリ変数を作成
        self.cnn_layers = OrderedDict() # 順番付きディクショナリ変数を初期化
        self.som_layers = OrderedDict() # 順番付きディクショナリ変数を初期化

        # Conv_Pooling層
        self.cnn_layers['Conv1'] = Convolution(filter_num, input_dim[0], filter_size, \
                                           conv_param['stride'], conv_param['pad'])
        self.cnn_layers['Relu1'] = Relu()
        self.cnn_layers['Pool1'] = Pooling(pool_h=2, pool_w=2)
        self.cnn_layers['Conv2'] = Convolution(filter_num, 12, filter_size, \
                                           conv_param['stride'], conv_param['pad'])
        self.cnn_layers['Relu2'] = Relu()
        self.cnn_layers['Pool2'] = Pooling(pool_h=2, pool_w=2)

        # Self_organizing層
        self.som_layers['SOM1'] = SOM(som_param['map_size'], som_param['alpha'], 2, som_param['input_vec'])
        self.som_layers['Tanh1'] = Tanh()

        self.loss_v = None


    def forward(self, x):
        for layer in self.cnn_layers.values():
            x = layer.forward(x)
        for layer in self.som_layers.values():
            x = layer.forward(x)

        return x

    # 自己組織化
    def self_organizing(self, x):

        for layer in self.cnn_layers.values():
            x = layer.forward(x)
        self.som_layers['SOM1'].self_organization(x)

    # CNNのパラメータ更新
    def update_param(self):
        # 勾配を計算する
        self.__gradient()

        # CNNのパラメータを更新
        self.cnn_layers['Conv1'].update_param()


    # 学習後のパラメータを書き出し
    def save_params(self, file_name="params.pkl"):
        # パラメータを格納
        params = {}
        params['Conv1'] =  self.cnn_layers['Conv1'].get_params()
        params['SOM1'] =  self.som_layers['SOM1'].get_params()

        # 書き出し
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    # 学習済みパラメータを読み込み
    def load_params(self, file_name="params.pkl"):
        # 読み込み
        with open(file_name, 'rb') as f:
            params = pickle.load(f)

        # インスタンス変数に格納
        for key, val in params.items():
            self.params[key] = val

        # 各レイヤのインスタンス変数に格納
        self.cnn_layers['Conv1'].set_params(params['Conv1'])
        self.som_layers['SOM1'].set_params(params['SOM1'])



     # 損失関数メソッドの定義
    def __loss(self):
        loss_value =  1 - self.som_layers['SOM1'].evaluation_function()
        self.loss_v = loss_value
        return np.full((self.som_input_size[0],self.som_input_size[1],self.som_input_size[2],self.som_input_size[3]), loss_value)

    # 勾配計算メソッドの定義
    def __gradient(self):
        # ニューロン非使用率を計算
        loss_vec = self.__loss()

        # 各レイヤを逆順に処理
        layers = list(self.cnn_layers.values())
        layers.reverse()
        for layer in layers:
            loss_vec = layer.backward(loss_vec)






if __name__ == '__main__':
    model = SelfOrganizingConvNet()
    mnist_data = mnist()
    mnist_data.Read_data(0)
    sample_data = []
    for i in range(10):
        sample_data.append(mnist_data.data_vec[i,7])
        sample_data[i]=sample_data[i].reshape(1,1,28,28)





    for step in tqdm(range(1000)):

        for class_num in range(10):
            # class_num =  np.random.randint(0,10)
            num =  0
            data = mnist_data.data_vec[class_num,num]
            data = data.reshape(1,1,28,28)
            model.self_organizing(data)

        # model.update_param()
        # print(f'step:{step} loss:{model.loss_v}')
    for i in range(10):
        fig = plt.figure()
        sns.heatmap(sample_data[i].reshape(28,28))
        fig.savefig(f"./out/mnist_{i}.png")

    for i in range(10):
        sample_data[i] = model.forward(sample_data[i])




    for i in range(10):
        fig = plt.figure()
        sns.heatmap(sample_data[i])
        fig.savefig(f"./out/img_{i}.png")


