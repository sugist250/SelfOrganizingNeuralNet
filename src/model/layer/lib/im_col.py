import numpy as np

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):

    # 入力データのサイズを取得
    N, C, H, W = input_data.shape

    # 出力データのサイズを計算
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    # パディング
    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')

    # 出力データの受け皿を初期化
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    # 行方向のインデックス
    for y in range(filter_h):
        # 行方向の最大値を計算
        y_max = y + stride * out_h

        # 列方向のインデックス
        for x in range(filter_w):
            # 列方向の最大値を計算
            x_max = x + stride * out_w

            # フィルターのy,x要素に対応する入力データの要素を抽出
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    # 出力サイズに整形
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col

def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):

    # 入力データのサイズを取得
    N, C, H, W = input_shape

    # 出力データのサイズを計算
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    # データとチャンネルに関する軸を分割
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    # 出力データの受け皿を初期化
    img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))

    # 行方向のインデックス
    for y in range(filter_h):
        # 行方向の最大値を計算
        y_max = y + stride * out_h

        # 列方向のインデックス
        for x in range(filter_w):
            # 列方向の最大値を計算
            x_max = x + stride * out_w

            # フィルターのy,x要素に対応する入力データの要素を抽出
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]
