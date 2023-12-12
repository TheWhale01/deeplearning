import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from log_loss_neuron import LogLossNeuron

# I don't know yet methods to find the most optimum learning rate
# and nb_iter, so I have to try out some values by hand

def main():
	# Making dataset
	X, y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=0)
	y = y.reshape((y.shape[0], 1))

	print(f"X dims: {X.shape}")
	print(f"y dims: {y.shape}")

	neuron = LogLossNeuron((X, y), 0.05, 500)
	neuron.train()
	neuron.show_learning_curve()

	
if __name__ == "__main__":
	main()
