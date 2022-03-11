from itertools import count
import sys
import os
import numpy as np
import csv
from collections import OrderedDict
import matplotlib.pyplot as plt
from tqdm import tqdm
from keras.datasets import mnist



sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.model.layer.affine import Afine
from src.model.layer.activation import Relu
from src.model.layer.soft_max import SoftmaxWithLoss
from data.mnist import Mnist

class Affine_Model():
    def __init__(self) -> None:

        self.layers = OrderedDict() # 順番付きディクショナリ変数を初期化

        # 第1層
        self.layers['affine1'] = Afine(28*28, 10)
        # self.layers['Relu1'] = Relu()
        # 第2層
        # self.layers['affine2'] = Afine(100, 10)
        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        # レイヤごとに順伝播の処理:(未正規化)
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def forward(self, x, t):
        for layer in self.layers.values():
            x = layer.forward(x)
        x = self.last_layer.forward(x, t)
        return x


    # パラメータ更新
    def update_param(self):
        # 勾配を計算する
        self.__gradient()

        # NNのパラメータを更新
        self.layers['affine1'].update_param()
        # self.layers['affine2'].update_param()


    def __loss(self):
        return self.last_layer.backward()

     # 勾配計算メソッドの定義
    def __gradient(self):
        # ニューロン非使用率を計算
        loss_vec = self.__loss()


        # 各レイヤを逆順に処理
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            loss_vec = layer.backward(loss_vec)



if __name__ == '__main__':
    affine_model = Affine_Model()
    m = Mnist()
    # mnist データをダウンロード
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    nomalize_train_images = train_images[:]/255
    nomalize_test_images = test_images[:]/255
    epoch = 1000
    loss_log = []
    for _ in tqdm(range(epoch)):
        for i in range(50):
                # class_num =  np.random.randint(0,10)
                # num = np.random.randint(0,100)
                t_data = np.array(m.retrun_onehot_vec(train_labels[i]))
                t_data = t_data.reshape(1,10)
                data = nomalize_train_images[i]
                data = data.reshape(1, 28*28)
                loss = affine_model.forward(data, t_data)

                affine_model.update_param()
        loss_log.append(loss)

    with open('./out/loss_log.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(loss_log)

    x = np.linspace(0, 1, epoch)
    # プロット
    plt.plot(x, loss_log, label="loss")

    # 凡例の表示
    plt.legend()
    plt.savefig("./out/loss.png")
    count = 0
    ans = np.zeros((10,10),dtype=int)
    for i in range(test_images.shape[0]):
        t_data = np.array(m.retrun_onehot_vec(test_labels[i]))
        t_data = t_data.reshape(1,10)
        data = nomalize_test_images[i]
        data = data.reshape(1, 28*28)
        predict = affine_model.predict(data)
        predict_num = np.argmax(predict)
        ans[predict_num][test_labels[i]] += 1
        print(f'pridict:{predict_num}, label:{test_labels[i]}')
        if predict_num == test_labels[i]:
            count += 1

    print(count/test_images.shape[0])
    for i in range(10):
        print(f'{i}:{ans[i]}')




