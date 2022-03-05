import imp


import numpy as np

# 交差エントロピー誤差の実装
def cross_entropy_error(y, t):
    # 2次元配列に変換
    if y.ndim == 1: # 1次元配列の場合
        # 1×Nの配列に変換
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 教師ラベルを取得
    if t.size == y.size: # one-hot表現の場合
        # データごとに最大値を抽出
        t = t.argmax(axis=1)

    # データ数を取得
    batch_size = y.shape[0]

    # 交差エントロピー誤差を計算:式(4.3)
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
