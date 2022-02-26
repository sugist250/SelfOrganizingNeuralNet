import numpy as np

class Relu():
    @staticmethod
    def forward(x):
        return np.maximum(0, x)

class Tanh():
    @staticmethod
    def forward(x):
        return np.tanh(x)


if __name__=='__main__':
    x = np.array([[1,2],[-1, -1]])
    t = Tanh()

    print(t.forward(x))
