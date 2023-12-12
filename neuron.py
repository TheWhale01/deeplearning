from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt

class Neuron(ABC):
    def __init__(self, dataset: tuple, learning_rate: float, nb_iter: int):
	    #X, y: dataset
	    #learning_rate: step for the gradient descent
	    #n_iter: the number of times that the model will train
        self.__X = dataset[0]
        self.y = dataset[1]
        self.__learning_rate = learning_rate
        self.__nb_iter = nb_iter
        self.__W = np.random.randn(self.__X.shape[1], 2)
        self.__b = np.random.randn(1)
        self.__losses = []
        self.__Z = None
        self.a = None
        self.__dW = None
        self.__db = None

    def model(self):
        self.__Z = self.__X.dot(self.__W) + self.__b
        self.a = 1 / (1 + np.exp(self.__Z * -1))

    def show_learning_curve(self):
        plt.plot(self.__losses)
        plt.show()

    def get_gradients(self):
        self.__dW = 1 / len(self.y) * np.dot(self.__X.T, self.a - self.y)
        self.__db = 1 / len(self.y) * np.sum(self.a - self.y)

    def update(self):
        self.__W -= (self.__learning_rate * self.__dW)
        self.__b -= (self.__learning_rate * self.__db)

    def train(self):
        for i in range(0, self.__nb_iter):
            self.model()
            self.__losses.append(self.loss())
            self.get_gradients()
            self.update()

    @abstractmethod
    def loss(self) -> float:
        pass
