"""
Training and analyzing networks.
"""

import os
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

import larq as lq
import tensorflow as tf
from time import time
import numpy as np

from matplotlib import pyplot as plt
plt.style.use("dark_background")

from data import Data
import settings

def plotCosts(bin_cnn_loss, cnn_loss, name):
    """Plots loss and accuracy as function of epochs to describe neural network training"""
    figs, axes = plt.subplots(1)
    axes.set_title("Training Evalutation: " + name)

    axes.plot(bin_cnn_loss, c="red", label="BinCNN")   
    axes.plot(cnn_loss, c="green", label="CNN")

    axes.set_ylabel("MSE")
    
    plt.xlabel("Epoch")

    axes.grid(alpha=0.3, ls="--")
    axes.legend()
    
def render_path(angles):
	"""Displays angle data as track representation. Result doesn't look right, but may not be directly tranlatable as hopded..."""
    mapping = {
        0: "red",
        1: "red",
        2: "red",
        3: "blue",
        4: "blue",
        5: "blue",
        6: "green",
        7: "green",        
        8: "green",
        9: "purple",
        10: "purple",
    }
    for limit in [i/5 for i in range(1, 6)]:
        plt.figure()
        pos = [(0, 0)]
        heading = np.pi / 2
        for i in range(angles.shape[0]-1):
            start = pos[i]
            angle = angles[i]
            heading -= limit * np.radians(angle)
            end = (start[0] + np.cos(heading), start[1] + np.sin(heading))
            pos.append(end)
            plt.plot((start[0], end[0]), (start[1], end[1]), color=mapping[(i%1000)//100])
        plt.title(str(limit))
    plt.show()
    
def train_and_evaluate(model, trainingData, trainingLabels, testingData, testingLabels, name="model", epochs=100, batch_size=128, verbose=True):
    """Trains and tests model"""
    print("-"*100)
    print("Evaluating and Evaluating " + name + ":")
    print()
    print("Conducting initial testing...")

    initialLoss = model.evaluate(testingData, testingLabels, verbose=0)
    print("Initial testing complete...")
    
    # training
    print("\nTraining...")
    timer = time()
    history = model.fit(trainingData, trainingLabels, batch_size=batch_size, epochs=epochs, verbose=verbose)
    elapsed = time() - timer

    print("\nTesting...")
    trainingLoss = model.evaluate(trainingData, trainingLabels, verbose=0)    
    testLoss = model.evaluate(testingData, testingLabels, verbose=verbose)

    print("\nTraining time: " + str(round(elapsed, 3)) + " secs")

    loss = history.history["loss"]
    print("-"*100)
    return loss
    
def BinCNN():
	"""Can experiment with model"""
    kwargs = {
        "input_quantizer": "ste_sign",
        "kernel_quantizer": "ste_sign",
        "kernel_constraint": "weight_clip",
        "use_bias": True,
    }

    model = tf.keras.models.Sequential()
    
    model.add(lq.layers.QuantConv2D(16, (3, 3),
                                        kernel_quantizer="ste_sign",
                                        kernel_constraint="weight_clip",
                                        activation='relu',
                                        input_shape=(*settings.img_size, 1)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.BatchNormalization(scale=False))

    model.add(lq.layers.QuantConv2D(32, (3, 3), activation='relu', **kwargs))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.BatchNormalization(scale=False))
    
    model.add(lq.layers.QuantConv2D(64, (3, 3), activation='relu', **kwargs))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.BatchNormalization(scale=False))
    model.add(tf.keras.layers.Flatten())

    model.add(lq.layers.QuantDense(500, activation='relu', **kwargs))
    model.add(tf.keras.layers.BatchNormalization(scale=False))
    
    model.add(lq.layers.QuantDense(100, activation='relu', **kwargs))
    model.add(tf.keras.layers.BatchNormalization(scale=False))
    
    model.add(lq.layers.QuantDense(50, activation='relu', **kwargs))
    model.add(tf.keras.layers.BatchNormalization(scale=False))
   
    model.add(lq.layers.QuantDense(10, activation='relu', **kwargs))
    model.add(tf.keras.layers.BatchNormalization(scale=False))

    model.add(lq.layers.QuantDense(1, **kwargs))

    # COMPILING
    model.compile(optimizer="adam",
                  loss='mse',
				  )
                  
    lq.models.summary(model)
    
    return model

def CNN():
	"""Can experiment with model"""
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(16, (3, 3), input_shape=(*settings.img_size, 1), activation='relu', use_bias=True))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    #model.add(tf.keras.layers.BatchNormalization(scale=False))

    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', use_bias=True))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    #model.add(tf.keras.layers.BatchNormalization(scale=False))
    
    #model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', use_bias=True))
    #model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    #model.add(tf.keras.layers.BatchNormalization(scale=False))
    model.add(tf.keras.layers.Flatten())

    #model.add(tf.keras.layers.Dense(500, activation='relu', use_bias=True))
    #model.add(tf.keras.layers.Dense(1164, activation='relu', use_bias=True))
    #model.add(tf.keras.layers.BatchNormalization(scale=False))
    
    model.add(tf.keras.layers.Dense(50, activation='relu', use_bias=True))
    #model.add(tf.keras.layers.BatchNormalization(scale=False))
    
   # model.add(tf.keras.layers.Dense(25, activation='relu', use_bias=True))
    #model.add(tf.keras.layers.BatchNormalization(scale=False))
   
    model.add(tf.keras.layers.Dense(10, activation='relu', use_bias=True))
    #model.add(tf.keras.layers.BatchNormalization(scale=False))

    model.add(tf.keras.layers.Dense(1, use_bias=True))

    # COMPILING
    model.compile(optimizer='adam',
                  loss='mse',
                  )
    return model
    
def main():
    #print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))
    #return
    print("\nLoading data...")
    data = Data()
    data.load_files()
    data.process_data()
    
    shuffled_data, shuffled_labels = data.get_shuffled_data(*data.get_data())
    
    m = shuffled_data.shape[0]
    num_training = int(0.9 * m)
    num_testing = m - num_training
    
    training_data, training_labels = shuffled_data[:num_training], shuffled_labels[:num_training]
    testing_data, testing_labels = shuffled_data[num_training:], shuffled_labels[num_training:]
    
    print("Training size:", num_training)
    print("Testing size:", num_testing)

    training_data = training_data.reshape((num_training, *settings.img_size, 1))
    testing_data = testing_data.reshape((num_testing, *settings.img_size, 1))
 
    print("Data loaded!")
    
    # render test
    render_path(training_labels)
    
    cwd = os.getcwd()
    model_path = os.path.join(cwd, "models")

    bin_cnn = BinCNN()
    cnn = CNN()

    # EVALUATING
    bin_cnn_loss = train_and_evaluate(bin_cnn, training_data, training_labels, testing_data, testing_labels, name="BinCNN")
    cnn_loss = train_and_evaluate(cnn, training_data, training_labels, testing_data, testing_labels, name="CNN")

    plotCosts(bin_cnn_loss, cnn_loss, "BinCNN vs CNN")
    
    train_predictions = cnn.predict(training_data)
    predictions = cnn.predict(testing_data)
    
    for i in range(200):
        print("Test:", testing_labels[i], round(predictions[i][0], 3), "        Train:", training_labels[i], round(train_predictions[i][0], 3))
    plt.show()
    
    print("\nDone!")

if __name__ == "__main__":
    main()