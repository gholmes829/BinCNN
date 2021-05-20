"""
MNIST handwritten digit classification with BinCNN. 
"""

import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
import larq as lq
from memory_profiler import profile

from time import time
from pympler import asizeof
from matplotlib import pyplot as plt

plt.style.use("dark_background")

def showImage(image):
	"""Displays image in grayscale"""
	plt.imshow(image, cmap="gray")

def plotCosts(loss, accuracy, name):
	"Plots loss and accuracy as function of epochs to describe neural network training"
	figs, axes = plt.subplots(2, sharex=True)

	plt.suptitle("MNIST Digit Classification with " + name)

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

def makeBinCNN():
	"""Create BinCNN"""
	kwargs = dict(input_quantizer="ste_sign",
              kernel_quantizer="ste_sign",
              kernel_constraint="weight_clip")

	model = tf.keras.models.Sequential()

	# Quantize the weights and not the input
	model.add(lq.layers.QuantConv2D(32, (3, 3),
										kernel_quantizer="ste_sign",
										kernel_constraint="weight_clip",
										use_bias=False,
										input_shape=(28, 28, 1)))

	model.add(tf.keras.layers.MaxPooling2D((2, 2)))
	model.add(tf.keras.layers.BatchNormalization(scale=False))

	model.add(lq.layers.QuantConv2D(64, (3, 3), use_bias=False, **kwargs))
	model.add(tf.keras.layers.MaxPooling2D((2, 2)))
	model.add(tf.keras.layers.BatchNormalization(scale=False))

	model.add(lq.layers.QuantConv2D(64, (3, 3), use_bias=False, **kwargs))
	model.add(tf.keras.layers.BatchNormalization(scale=False))
	model.add(tf.keras.layers.Flatten())

	model.add(lq.layers.QuantDense(64, use_bias=False, **kwargs))
	model.add(tf.keras.layers.BatchNormalization(scale=False))

	model.add(lq.layers.QuantDense(10, use_bias=False, **kwargs))
	model.add(tf.keras.layers.BatchNormalization(scale=False))

	model.add(tf.keras.layers.Activation("softmax"))

	# compiling

	model.compile(optimizer='adam',
				  loss='sparse_categorical_crossentropy',
				  metrics=['accuracy'])

	return model

def makeCNN():
	"""Create CNN"""
	model = tf.keras.models.Sequential()

	model.add(tf.keras.layers.Conv2D(32, (3, 3), use_bias=False, input_shape=(28, 28, 1)))

	model.add(tf.keras.layers.MaxPooling2D((2, 2)))
	model.add(tf.keras.layers.BatchNormalization(scale=False))

	model.add(tf.keras.layers.Conv2D(64, (3, 3), use_bias=False))
	model.add(tf.keras.layers.MaxPooling2D((2, 2)))
	model.add(tf.keras.layers.BatchNormalization(scale=False))

	model.add(tf.keras.layers.Conv2D(64, (3, 3), use_bias=False))
	model.add(tf.keras.layers.BatchNormalization(scale=False))
	model.add(tf.keras.layers.Flatten())

	model.add(tf.keras.layers.Dense(64, use_bias=False))
	model.add(tf.keras.layers.BatchNormalization(scale=False))

	model.add(tf.keras.layers.Dense(10, use_bias=False))
	model.add(tf.keras.layers.BatchNormalization(scale=False))

	model.add(tf.keras.layers.Activation("softmax"))

	# compiling
	model.compile(optimizer='adam',
				  loss='sparse_categorical_crossentropy',
				  metrics=['accuracy'])

	return model

def train_and_evaluate(model, trainingData, trainingLabels, testingData, testingLabels, name="model", epochs=8, batch_size=128, verbose=1):
	"""Trains and tests model"""
	print("-"*100)
	print("Evaluating and Evaluating " + name + ":")
	print()
	print("Initial testing...")
	initialLoss, initialAccuracy = model.evaluate(testingData, testingLabels, verbose=verbose)

	# training
	print("Training...")
	timer = time()
	history = model.fit(trainingData, trainingLabels, batch_size=batch_size, epochs=epochs, verbose=verbose)
	elapsed = time() - timer

	print("Testing...")
	trainingLoss, trainingAccuracy = model.evaluate(trainingData, trainingLabels, verbose=verbose)	
	testLoss, testAccuracy = model.evaluate(testingData, testingLabels, verbose=verbose)

	print("\nTraining time: " + str(round(elapsed, 3)) + " secs")

	print("\nIniital accuracy: " + str(round(100 * initialAccuracy, 3)))
	print("Training accuracy: " + str(round(100 * trainingAccuracy, 3)))
	print("Test accuracy: " + str(round(100 * testAccuracy, 3)))

	loss = np.array([initialLoss] + history.history["loss"])
	accuracy = np.array([initialAccuracy] + history.history["accuracy"])
	print("-"*100)
	return loss, accuracy
	
@profile
def feedForwardBinCNN(model, data, labels):
	model.evaluate(data, labels, verbose=0)

@profile
def feedForwardCNN(model, data, labels):
	model.evaluate(data, labels, verbose=0)

def main():
	print("BINCNN AND CNN DIGIT RECOGNITION TESTING\n")
	
	print("-"*100)
	print("Loading data:")
	dataScale = 1  # in [0, 1], determines what percent of data should be used

	(trainingData, trainingLabels), (testingData, testingLabels) = tf.keras.datasets.mnist.load_data()

	trainingData = trainingData.reshape((60000, 28, 28, 1))
	testingData = testingData.reshape((10000, 28, 28, 1))

	trainingSize, testingSize = trainingData.shape[0], testingData.shape[0]
	scaledTrainingSize, scaledTestingSize = int(dataScale * trainingSize), int(dataScale * testingSize)
	
	trainingData, trainingLabels = trainingData[:scaledTrainingSize], trainingLabels[:scaledTrainingSize]
	testingData, testingLabels = testingData[:scaledTestingSize], testingLabels[:scaledTestingSize]

	showImage(np.squeeze(trainingData[0]))  # example of data

	# Normalize pixel values to be between -1 and 1
	trainingData, testingData = trainingData / 127.5 - 1, testingData / 127.5 - 1

	print("\nData loaded!\n")

	print("Using " + str(trainingData.shape[0]) + " training examples")
	print("Using " + str(testingData.shape[0]) + " testing examples")
	print("-"*100)
	print("\n")
	
	bin_cnn = makeBinCNN()
	cnn = makeCNN()

	bin_cnn_loss, bin_cnn_accuracy = train_and_evaluate(bin_cnn, trainingData, trainingLabels, testingData, testingLabels, name="BinCNN")
	print("\n")
	cnn_loss, cnn_accuracy = train_and_evaluate(cnn, trainingData, trainingLabels, testingData, testingLabels, name="CNN")

	plotCosts(bin_cnn_loss, bin_cnn_accuracy, "BinCNN")
	plotCosts(cnn_loss, cnn_accuracy, "CNN")

	bin_cnn_size = asizeof.asizeof(bin_cnn)
	cnn_size = asizeof.asizeof(cnn)

	print("\n\n" + "-"*100)
	print("Profiling size of objects...\n")
	print("Size of BinCNN: " + str(bin_cnn_size) + " bytes")
	print("Size of CNN: " + str(cnn_size) + " bytes")
	print("-"*100)
	print("\n")

	# memory
	print("-"*100)
	print("Profiling memory:\n")
	feedForwardBinCNN(bin_cnn, testingData, testingLabels)
	feedForwardCNN(cnn, testingData, testingLabels)
	print("-"*100)
	print("\n")

	# time
	print("-"*100)
	print("Profiling time:\n")
	n = 25

	timer = time()
	for _ in range(n):
		bin_cnn.evaluate(testingData, testingLabels, verbose=0)
	elapsed = time() - timer	
	print("Avg BinCNN feed forward time for n=" + str(n) + ": " + str(round(elapsed/n, 3)) + " secs")

	timer = time()
	for _ in range(n):
		cnn.evaluate(testingData, testingLabels, verbose=0)
	elapsed = time() - timer	
	print("Avg CNN feed forward time for n=" + str(n) + ": " + str(round(elapsed/n, 3)) + " secs")
	print("-"*100)

	print()
	plt.show()  # display neural network performance

if __name__ == "__main__":
	main()

