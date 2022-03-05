import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.model.layer.activation import Soft_Max
from src.model.layer.lib.cross_entropy import cross_entropy_error

# Softmax-with-Lossレイヤの実装
class SoftmaxWithLoss:
    # 初期化メソッド
    def __init__(self):
        # 変数を初期化
        self.loss = None # 交差エントロピー誤差
        self.y = None # ニューラルネットワークの出力
        self.t = None # 教師ラベル

        self.softmax = Soft_Max()

    # 順伝播メソッド
    def forward(self, x, t):
        # 教師ラベルを保存
        self.t = t

        # ソフトマックス関数による活性化(正規化)
        self.y = self.softmax.forward(x)

        # 交差エントロピー誤差を計算
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    # 逆伝播メソッド
    def backward(self, dout=1):
        # バッチサイズを取得
        batch_size = self.t.shape[0]

        # 順伝播の入力の勾配を計算
        dx = (self.y - self.t) / batch_size
        return dx
