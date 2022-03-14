import sys
import os
from collections import OrderedDict

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.model.layer.affine import Afine
from src.model.layer.activation import Relu
from src.model.layer.soft_max import SoftmaxWithLoss
from src.model.layer.convolution import Convolution
from src.model.layer.pooling import Pooling


class CA_Model():
    def __init__(self) -> None:
        # 順番付きディクショナリ変数を初期化
        self.cnn_layers =  OrderedDict()
        self.affine_layers = OrderedDict()


        # Conv_Pooling層
        self.cnn_layers['Conv1'] = Convolution(filter_num=4, input_dim=1, filter_size=2,stride=2)
        self.cnn_layers['Relu1'] = Relu()
        self.cnn_layers['Pool1'] = Pooling(pool_h=2, pool_w=2)
        self.cnn_layers['Conv2'] = Convolution(filter_num=8, input_dim=4, filter_size=3,stride=2)
        self.cnn_layers['Relu2'] = Relu()
        self.cnn_layers['Pool2'] = Pooling(pool_h=2, pool_w=2)


        # 全結合層
        self.affine_layers['affine1'] = Afine(25*8, 10)
        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        # レイヤごとに順伝播の処理:(未正規化)
        for cnn_layer in self.cnn_layers.values():
            x = cnn_layer.forward(x)

        for affine_layer in self.affine_layers.values():
            x = affine_layer.forward(x)

        return x

    def forward(self, x, t):
        # レイヤごとに順伝播の処理:(未正規化)
        for cnn_layer in self.cnn_layers.values():
            x = cnn_layer.forward(x)

        for affine_layer in self.affine_layers.values():
            x = affine_layer.forward(x)

        x = self.last_layer.forward(x, t)
        return x


    # パラメータ更新
    def update_param(self):
        # 勾配を計算する
        self.__gradient()

        # NNのパラメータを更新
        self.affine_layers['affine1'].update_param()


    def __loss(self):
        # 損失関数
        return self.last_layer.backward()

    def __gradient(self):
        #勾配を計算
        loss_vec = self.__loss()
        layers = list(self.affine_layers.values())
        layers.reverse()
        for layer in layers:
            loss_vec = layer.backward(loss_vec)


if __name__ == '__main__':
    pass
