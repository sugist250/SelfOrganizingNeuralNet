from turtle import update
import numpy as np
# Adamの実装
class Adam:

    # インスタンス変数を定義
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999):
        self.lr = lr # 学習率
        self.beta1 = beta1 # mの減衰率
        self.beta2 = beta2 # vの減衰率
        self.iter = 0 # 試行回数を初期化
        self.m = None # モーメンタム
        self.v = None # 適合的な学習係数

    # パラメータの更新メソッドを定義
    def update(self, param, grads):
        updated_param = param
        # mとvを初期化
        if self.m is None: # 初回のみ
            self.m = np.zeros_like(param) # 全ての要素が0
            self.v = np.zeros_like(param) # 全ての要素が0

        # パラメータごとに値を更新
        self.iter += 1 # 更新回数をカウント
        lr_t  = self.lr * np.sqrt(1.0 - self.beta2 ** self.iter) / (1.0 - self.beta1 ** self.iter) # 式(6)の学習率の項

        self.m = self.beta1 * self.m + (1 - self.beta1) * grads # 式(1)
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads ** 2) # 式(2)
        updated_param -= lr_t * self.m / (np.sqrt(self.v) + 1e-7) # 式(6)

        return updated_param

if __name__ == '__main__':
    adam = Adam()
    def function(x):
        y = x**2
        return y
    def d_function(x):
        dy = 2*x
        return dy

    x = np.array([10.0])
    dy =  d_function(np.array([10.0]))
    print(x, dy)
    for _ in range(50000):
        # パラメータを更新
        x = adam.update(x, dy)
        dy =  d_function(x)
        print(x, dy)

