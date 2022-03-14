import sys
import os
import numpy as np


sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from data.mnist import Mnist
from src.model.layer.lib.adam import Adam
from src.model.layer.lib.im_col import im2col, col2im

class Convolution:

    # インスタンス変数の定義
    def __init__(self, filter_num,  input_dim, filter_size, stride=1, pad=0):
        self.weight_init_std=0.01
        self.W = self.weight_init_std * np.random.randn(filter_num, input_dim, filter_size, filter_size)
        self.b = np.zeros(filter_num)
        self.stride = stride # ストライド
        self.pad = pad # パディング

        # (逆伝播時に使用する)中間データを初期化
        self.x = None # 入力データ
        self.col = None # 2次元配列に展開した入力データ
        self.col_W = None # 2次元配列に展開したフィルター(重み)

        # 勾配に関する変数を初期化
        self.dW = None # フィルター(重み)に関する勾配
        self.db = None # バイアスに関する勾配

        # 最適化手法のインスタンスを作成
        self.optimizer_w = Adam(lr=0.001)
        self.optimizer_b = Adam(lr=0.001)

    # 順伝播メソッドの定義
    def forward(self, x):
        # 各データに関するサイズを取得
        FN, C, FH, FW = self.W.shape # フィルター
        N, C, H, W = x.shape # 入力データ
        out_h = int(1 + (H + 2 * self.pad - FH) / self.stride) # 出力データ:式(7.1)
        out_w = int(1 + (W + 2 * self.pad - FW) / self.stride)

        # 各データを2次元配列に展開
        col = im2col(x, FH, FW, self.stride, self.pad) # 入力データ
        col_W = self.W.reshape(FN, -1).T # フィルター

        # 出力の計算:(図7-12)
        out = np.dot(col, col_W) + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        # (逆伝播時に使用する)中間データを保存
        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    # 逆伝播メソッドの定義
    def backward(self, dout):
        # フィルターに関するサイズを取得
        FN, C, FH, FW = self.W.shape

        # 順伝播の入力を展開
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)

        # 各パラメータの勾配を計算:式(5.13)
        self.db = np.sum(dout, axis=0) # バイアス
        self.dW = np.dot(self.col.T, dout) # (展開した)重み
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW) # 本来の形状に変換
        dcol = np.dot(dout, self.col_W.T) # (展開した)入力データ
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad) # 本来の形状に変換

        return dx

    def update_param(self):
        self.W = self.optimizer_w.update(self.W, self.dW)
        self.b = self.optimizer_b.update(self.b, self.db)




if __name__ == '__main__':
    b = np.zeros(1)
    W = np.random.rand(30,1,3,3)
    conv = Convolution(W,b,pad=1)
    mnist_data = mnist()
    mnist_data.Read_data(0)
    sample_data_0 = mnist_data.data_vec[0,1]
    sample_data_0 = sample_data_0.reshape(1,1,14,14)
    print(sample_data_0.shape)
    conv_data = conv.forward(sample_data_0)
    print(conv_data.shape)

