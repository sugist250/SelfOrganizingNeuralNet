import sys
import os
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from src.model.layer.lib.adam import Adam

class Afine():
    def __init__(self, inputs_size, outputs_size) -> None:
        # パラメータの初期化
        self.weight_init_std=0.01
        self.w =  self.weight_init_std * np.random.randn(inputs_size, outputs_size)
        self.b = np.zeros(outputs_size)

        #入力と勾配を初期化
        self.x = None  # 順伝播の入力
        self.dW = None # 重みの勾配
        self.db = None # バイアスの勾配
        self.original_x_shape = None

        # 最適化手法のインスタンスを作成
        self.optimizer_w = Adam(lr=0.001)
        self.optimizer_b = Adam(lr=0.001)

    def forward(self, x):
        # 入力データのサイズを保存
        self.original_x_shape = x.shape

        # バッチサイズ行の2次元配列に変形
        x = x.reshape(x.shape[0], -1)
        self.x = x
        out = np.dot(x, self.w) + self.b
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.w.T)      # 順伝播の入力の勾配
        self.dW = np.dot(self.x.T, dout) # 重みの勾配
        self.db = np.sum(dout, axis=0)   # バイアスの勾配

        return dx

    def update_param(self):
        self.w = self.optimizer_w.update(self.w, self.dW)
        self.b = self.optimizer_b.update(self.b, self.db)

if __name__ == '__main__':
    layer = Afine(50, 10)
    print(layer.x)
    print(layer.w.shape)
    print(layer.b.shape)
    X = np.random.randn(1,50)
    Y = layer.forward(X)
    print(Y.shape)
    # 逆伝播を計算
    dY = np.random.randn(1,10)
    dX = layer.backward(dY)
    print(dX.shape)
    print(layer.dW.shape)
    print(layer.db.shape)


