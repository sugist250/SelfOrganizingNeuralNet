import csv
import os
import matplotlib.pyplot as plt
import numpy as np

class Log():
    def __init__(self) -> None:
        # 損失のログ
        self.loss_log = []

        self.accuracy_log = []
        self.average_recall_log = []
        self.average_precision_log = []
        self.average_F_value_A_log = []
        self.average_F_value_B_log = []

    def add_log(self, loss, accuracy, average_recall, average_precision, average_F_value_A, average_F_value_B):

        self.loss_log.append(loss)
        self.accuracy_log.append(accuracy)
        self.average_recall_log.append(average_recall)
        self.average_precision_log.append(average_precision)
        self.average_F_value_A_log.append(average_F_value_A)
        self.average_F_value_B_log.append(average_F_value_B)

    def export_csv(self, dir_name):
        if not os.path.exists(f'out/{dir_name}'):
            os.mkdir(f'out/{dir_name}')

        with open(f'./out/{dir_name}/log.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(self.loss_log)
            writer.writerow(self.accuracy_log)
            writer.writerow(self.average_recall_log)
            writer.writerow(self.average_precision_log)
            writer.writerow(self.average_F_value_A_log)
            writer.writerow(self.average_F_value_B_log)

    def export_graph(self,  dir_name):
        if not os.path.exists(f'out/{dir_name}'):
            os.mkdir(f'out/{dir_name}')

        x = np.linspace(0, 1, len(self.loss_log))

        fig = plt.figure()
        fig.plot(x, self.loss_log, label="loss")
        plt.legend('loss')
        fig.savefig(f"./out/{dir_name}/loss.png")

        fig = plt.figure()
        fig.plot(x, self.accuracy_log, label="accuracy")
        plt.legend('accuracy')
        fig.savefig(f"./out/{dir_name}/accuracy.png")

        fig = plt.figure()
        fig.plot(x, self.average_recall_log, label="average_recall")
        plt.legend('average_recall')
        fig.savefig(f"./out/{dir_name}/average_recall.png")

        fig = plt.figure()
        fig.plot(x, self.average_precision_log, label="average_precision")
        plt.legend('average_precision')
        fig.savefig(f"./out/{dir_name}/average_precision.png")

        fig = plt.figure()
        fig.plot(x, self.average_F_value_A_log, label="average_F_value_A")
        plt.legend('average_F_value_A')
        fig.savefig(f"./out/{dir_name}/average_F_value_A.png")

        fig = plt.figure()
        fig.plot(x, self.average_F_value_B_log, label="average_F_value_B")
        plt.legend('average_F_value_B')
        fig.savefig(f"./out/{dir_name}/average_F_value_B.png")




