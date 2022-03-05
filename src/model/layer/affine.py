import sys
import os
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from data.mnist import mnist
from src.model.layer.lib.adam import Adam

class Afine():
    def __init__(self, inputs_size) -> None:
        # パラメータの初期化
        self.w =  self.weight_init_std * np.random.randn(inputs_size)
        self.b = np.zeros(inputs_size)

        #入力と勾配を初期化
        self.x = None  # 順伝播の入力
        self.dW = None # 重みの勾配
        self.db = None # バイアスの勾配

        # 最適化手法のインスタンスを作成
        self.optimizer_w = Adam(lr=0.001)
        self.optimizer_b = Adam(lr=0.001)

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)      # 順伝播の入力の勾配
        self.dW = np.dot(self.x.T, dout) # 重みの勾配
        self.db = np.sum(dout, axis=0)   # バイアスの勾配

        return dx

    def update_param(self):
        self.W = self.optimizer_w.update(self.W, self.dW)
        self.b = self.optimizer_b.update(self.b, self.db)


