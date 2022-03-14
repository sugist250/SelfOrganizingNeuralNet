
from numpy import dtype

import numpy as np

class Evaluation():
    def __init__(self, class_num) -> None:
        self.confusion_matrix = np.zeros((class_num, class_num), dtype=int)

    def add_confusion_matrix(self, predict, label):
        self.confusion_matrix[label][predict] += 1

    def init_confusion_matrix(self):
        self.confusion_matrix.fill(0)

    def make_test_confusion_matrix(self):
        test = [[240, 20, 40], [5, 180, 15], [25, 5 ,70]]
        for i in range(3):
            for j in range(3):
                self.confusion_matrix[i][j] = test[i][j]

    def evaluation_model(self):
        class_sum_value = np.sum(self.confusion_matrix, axis=1)
        predict_sum_value = np.sum(self.confusion_matrix, axis=0)
        diag = np.diag(self.confusion_matrix)
        F_value = ((diag/class_sum_value) + (diag/predict_sum_value)) / 2
        # 正解率
        accuracy = np.sum(diag)/np.sum(class_sum_value)

        # 再現率の平均
        average_recall = np.mean(diag/class_sum_value)

        # 適合率の平均
        average_precision = np.mean(diag/predict_sum_value)

        # F値の平均（定義A）
        average_F_value_A = np.mean(F_value)

        # F値の平均（定義B）
        average_F_value_B = (average_recall + average_precision)/2

        return accuracy, average_recall, average_precision, average_F_value_A, average_F_value_B



if __name__ == '__main__':
    evalution = Evaluation(3)
    print(evalution.confusion_matrix)
    evalution.make_test_confusion_matrix()
    print(evalution.confusion_matrix)

    print(evalution.evaluation_model())

    evalution.init_confusion_matrix()
    print(evalution.confusion_matrix)





