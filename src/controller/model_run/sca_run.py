import sys
import os
import numpy as np
from tqdm import tqdm
from keras.datasets import mnist
import matplotlib.pyplot as plt
import seaborn as sns




sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from src.model.SCA_model import SCA_Model
from controller.lib.log import Log
from controller.lib.evaluation import Evaluation
from src.model.layer.activation import Sigmoid, Tanh, Relu
from data.mnist import Mnist


def sca_model_run(epoch=100):
    r = Relu()
    # CAモデルを作成
    sca_model = SCA_Model()
    # mnistデータセット
    m = Mnist()
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    # mnistを正規化
    # label = []
    # t_images_n = []
    # for l in range(10):
    #     label.append(np.where(train_labels==l))
    #     t_images_n.append(train_images[label[l][0][0:10]])

    # t_images = []
    # t_labels = []
    # for n in range(10):
    #     for l in range(10):
    #         t_images.append(t_images_n[l][n,:,:])
    #         t_labels.append(train_labels[label[l][0][n]])
    # t_images = np.array(t_images)

    nomalize_train_images = train_images[:]/255
    nomalize_test_images = test_images[:]/255
    # train_data_num = train_images.shape[0]
    train_data_num = 100
    # ログインスタンスを生成
    train_log = Log()
    test_log = Log()
    # 評価インスタンスを生成
    train_evalution = Evaluation(10)
    test_evalution = Evaluation(10)


    print('self organizing......')
    for _ in tqdm(range(5)):
        for i in range(train_data_num):
                t_data = np.array(m.retrun_onehot_vec(train_labels[i]))
                t_data = t_data.reshape(1,10)
                data = nomalize_train_images[i]
                data = data.reshape(28,28)
                sca_model.self_organizing(data)

    weight_data = []
    for i in range(1000):
        data = nomalize_train_images[i]
        data = data.reshape(28,28)
        weight_data.append(sca_model.som_layers['SOM1'].forward(data))

    for i in range(10):
        fig = plt.figure()
        sns.heatmap(r.forward(weight_data[i]))
        fig.savefig(f"./out/img_{i}.png")

    for _ in tqdm(range(epoch)):
        # ロスの合計値の初期化
        sum_loss = 0
        # 混合行列の初期化
        train_evalution.init_confusion_matrix()
        test_evalution.init_confusion_matrix()

        # 学習
        for i in range(train_data_num):
            t_data = np.array(m.retrun_onehot_vec(train_labels[i]))
            t_data = t_data.reshape(1,10)
            data = nomalize_train_images[i]
            data = data.reshape(1,1,28,28)
            sum_loss += sca_model.forward(data, t_data)

            # パラメータの更新
            sca_model.update_param()

            # 予測
            predict = sca_model.predict(data)
            predict_num = np.argmax(predict)
            # 混合行列
            train_evalution.add_confusion_matrix(predict_num, train_labels[i])


        # モデルの評価
        average_loss = sum_loss/train_data_num
        accuracy, average_recall, average_precision, average_F_value_A, average_F_value_B = train_evalution.evaluation_model()
        train_log.add_log(average_loss, accuracy, average_recall, average_precision, average_F_value_A, average_F_value_B)


        # 評価
        for i in range(test_images.shape[0]):
            t_data = np.array(m.retrun_onehot_vec(test_labels[i]))
            t_data = t_data.reshape(1,10)
            data = nomalize_test_images[i]
            data = data.reshape(1,1,28,28)

            # 予測
            predict = sca_model.predict(data)
            predict_num = np.argmax(predict)
            # 混合行列
            test_evalution.add_confusion_matrix(predict_num, test_labels[i])

         # モデルの評価
        average_loss = sum_loss/train_data_num
        accuracy, average_recall, average_precision, average_F_value_A, average_F_value_B = test_evalution.evaluation_model()
        test_log.add_log(average_loss, accuracy, average_recall, average_precision, average_F_value_A, average_F_value_B)

    train_log.export_csv('sca_model_train')
    test_log.export_csv('sca_model_test')

    # weight_data = []
    # for i in range(10):
    #     data = nomalize_train_images[i]
    #     data = data.reshape(1,1,28,28)
    #     weight_data.append(sca_model.som_layers['SOM1'].forward(data))





    # for i in range(10):
    #     fig = plt.figure()
    #     sns.heatmap(r.forward(weight_data[i]),vmin=-1.0, vmax=1.0)
    #     fig.savefig(f"./out/img_{i}.png")




if __name__ == '__main__':
    sca_model_run(epoch=100)
