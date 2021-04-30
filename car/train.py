"""

"""

import os

import larq as lq
from matplotlib import pyplot as plt

from data import Data

def plotCosts(loss, accuracy, name):
	"""Plots loss and accuracy as function of epochs to describe neural network training"""
	figs, axes = plt.subplots(2, sharex=True)

	plt.suptitle("Training Evalutation: " + name)

	axes[0].plot(accuracy, c="cyan")
	axes[0].plot(accuracy, "o", c="cyan")

	axes[0].set_ylabel("Percent Correct")
	axes[0].set_title("Accuracy")

	axes[1].plot(loss, c="red")
	axes[1].plot(loss, "o", c="red")


	axes[1].set_ylabel("RMSE")
	axes[1].set_title("Loss")
	
	plt.xlabel("Epoch")

	axes[0].grid(alpha=0.25, ls="--")
	axes[1].grid(alpha=0.25, ls="--")

def main():
    print("Loading data...")
    data = Data()
    data.load_files()
    data.process_data()
    episodes = data.get_episodes()
    
    num_episodes = len(episodes)
    
    training_data, training_labels = data.get_collapsed_data(episodes[:7])
    testing_data, testing_labels = data.get_collapsed_data(episodes[7:])
    m = features.shape[0]
    print("Data loaded!")

    cwd = os.getcwd()
    model_path = os.path.join(cwd, "models")
    

if __name__ == "__main__":
    main()