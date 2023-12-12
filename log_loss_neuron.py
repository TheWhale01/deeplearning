from neuron import Neuron
import numpy as np

class LogLossNeuron(Neuron):
    def loss(self) -> float:
        return 1 / len(self.y) * np.sum((self.y * -1) * np.log(self.a) - (1 - self.y) * np.log(1 - self.a))
