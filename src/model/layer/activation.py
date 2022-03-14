import numpy as np

class Relu():
    # 初期化メソッド
    def __init__(self):
        # 順伝播の入力における0以下の要素の情報を初期化
        self.mask = None

    # 順伝播メソッド
    def forward(self, x):
        # 0以下の要素の情報を保存
        self.mask = (x <= 0)

        # 順伝播の入力を複製
        out = x.copy()

        # 0以下の要素を0に置換
        out[self.mask] = 0
        return out

    # 逆伝播メソッド
    def backward(self, dout):
        # 順伝播時に0以下だった要素を0に置換
        dout[self.mask] = 0

        # 複製
        dx = dout
        return dx


class Tanh():
    @staticmethod
    def forward(x):
        return np.tanh(x)

class Soft_Max():
    # ソフトマックス関数を実装
    @staticmethod
    def  forward(x):
        # 式(3.10)の計算
        c = np.max(x) # 最大値
        exp_a = np.exp(x - c) # 分子:オーバーフロー対策
        sum_exp_a = np.sum(exp_a) # 分母
        out = exp_a / sum_exp_a # 式(3.10)
        return out

if __name__=='__main__':
    x = np.array([[1,2],[-1, -1]])
    t = Tanh()

    print(t.forward(x))
