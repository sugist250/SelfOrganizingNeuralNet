import sys
import os
import numpy as np
from tqdm import tqdm
from keras.datasets import mnist
import matplotlib.pyplot as plt
import seaborn as sns




sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from src.model.CSA_model_copy import CSA_Model_CP
from controller.lib.log import Log
from controller.lib.evaluation import Evaluation
from src.model.layer.activation import Sigmoid, Tanh, Relu
from data.mnist import Mnist


def csa_model_run(epoch=100):
    r = Relu()
    # CAモデルを作成
    csa_model = CSA_Model_CP()
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
    train_data_num = 10000
    # ログインスタンスを生成
    train_log = Log()
    test_log = Log()
    # 評価インスタンスを生成
    train_evalution = Evaluation(10)
    test_evalution = Evaluation(10)

    for _ in range(10):
        print('self organizing......')
        for i in tqdm(range(train_data_num)):
                t_data = np.array(m.retrun_onehot_vec(train_labels[i]))
                t_data = t_data.reshape(1,10)
                data = nomalize_train_images[i]
                data = data.reshape(1,1,28,28)
                csa_model.self_organizing(data)



        weight_data = []
        for i in range(10):
            data = nomalize_train_images[i]
            data = data.reshape(1,1,28,28)
            for layer in csa_model.cnn_layers.values():
                data = layer.forward(data)
                csa_model.som_layers['SOM1'].forward(data))

        for i in range(10):
            fig = plt.figure()
            sns.heatmap(r.forward(weight_data[i]))
            fig.savefig(f"./out/img_{i}.png")

        csa_model.update_param()

    # for _ in tqdm(range(epoch)):
    #     # ロスの合計値の初期化
    #     sum_loss = 0
    #     # 混合行列の初期化
    #     train_evalution.init_confusion_matrix()
    #     test_evalution.init_confusion_matrix()

    #     # 学習
    #     for i in range(train_data_num):
    #         t_data = np.array(m.retrun_onehot_vec(t_labels[i]))
    #         t_data = t_data.reshape(1,10)
    #         data = nomalize_train_images[i]
    #         data = data.reshape(1,1,28,28)
    #         sum_loss += csa_model.forward(data, t_data)

    #         # パラメータの更新
    #         csa_model.update_param()

    #         # 予測
    #         predict = csa_model.predict(data)
    #         predict_num = np.argmax(predict)
    #         # 混合行列
    #         train_evalution.add_confusion_matrix(predict_num, t_labels[i])


    #     # モデルの評価
    #     average_loss = sum_loss/train_data_num
    #     accuracy, average_recall, average_precision, average_F_value_A, average_F_value_B = train_evalution.evaluation_model()
    #     train_log.add_log(average_loss, accuracy, average_recall, average_precision, average_F_value_A, average_F_value_B)


    #     # 評価
    #     for i in range(test_images.shape[0]):
    #         t_data = np.array(m.retrun_onehot_vec(test_labels[i]))
    #         t_data = t_data.reshape(1,10)
    #         data = nomalize_test_images[i]
    #         data = data.reshape(1,1,28,28)

    #         # 予測
    #         predict = csa_model.predict(data)
    #         predict_num = np.argmax(predict)
    #         # 混合行列
    #         test_evalution.add_confusion_matrix(predict_num, test_labels[i])

    #      # モデルの評価
    #     average_loss = sum_loss/train_data_num
    #     accuracy, average_recall, average_precision, average_F_value_A, average_F_value_B = test_evalution.evaluation_model()
    #     test_log.add_log(average_loss, accuracy, average_recall, average_precision, average_F_value_A, average_F_value_B)

    # train_log.export_csv('csa_model_train_f')
    # test_log.export_csv('csa_model_test_f')

    # weight_data = []
    # for i in range(10):
    #     data = nomalize_train_images[i]
    #     data = data.reshape(1,1,28,28)
    #     weight_data.append(csa_model.som_layers['SOM1'].forward(data))





    for i in range(10):
        fig = plt.figure()
        sns.heatmap(r.forward(weight_data[i]),vmin=-1.0, vmax=1.0)
        fig.savefig(f"./out/img_{i}.png")




if __name__ == '__main__':
    csa_model_run(epoch=10)
