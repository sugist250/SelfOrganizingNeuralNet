import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from data.mnist import mnist
from im_col import im2col, col2im

class Pooling:

    # インスタンス変数の定義
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h # Poolingの高さ
        self.pool_w = pool_w # Poolingの横幅
        self.stride = stride # ストライド
        self.pad = pad # パディング

        # 逆伝播用の中間データ
        self.x = None # 入力データ
        self.arg_max = None # 最大値のインデックス

    # 順伝播メソッドの定義
    def forward(self, x):
        # 各データに関するサイズを取得
        N, C, H, W = x.shape # 入力サイズ
        out_h = int(1 + (H - self.pool_h) / self.stride) # 出力サイズ:式(7.1)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        # 入力データを2次元配列に展開
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h * self.pool_w)

        # 逆伝播用に最大値のインデックスを保存
        arg_max = np.argmax(col, axis=1)

        # 出力データを作成
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2) # 4次元配列に再変換:(図7-20)

        # 逆伝播用に中間データを保存
        self.x = x # 入力データ
        self.arg_max = arg_max # 最大値のインデックス

        return out

    # 逆伝播メソッドの定義
    def backward(self, dout):
        # 入力データを変形:(図7-20の逆方法)
        dout = dout.transpose(0, 2, 3, 1)

        # 受け皿を作成
        pool_size = self.pool_h * self.pool_w # Pooling適用領域の要素数
        dmax = np.zeros((dout.size, pool_size)) # 初期化

        # 最大値の要素のみ伝播
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()

        # 4次元配列に変換
        dmax = dmax.reshape(dout.shape + (pool_size,))
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)

        return dx

if __name__ == '__main__':
    b = np.zeros(1)
    pool_w = 2
    pool_h = 2
    pool = Pooling(pool_h,pool_w)
    mnist_data = mnist()
    mnist_data.Read_data(0)
    sample_data_0 = mnist_data.data_vec[0,1]
    sample_data_0 = sample_data_0.reshape(1,1,14,14)
    print(sample_data_0.shape)
    conv_data = pool.forward(sample_data_0)
    print(conv_data.shape)
