"""

"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import larq as lq
import tensorflow as tf
from time import time
import numpy as np

from matplotlib import pyplot as plt
plt.style.use("dark_background")

from data import Data
import settings

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
    
def train_and_evaluate(model, trainingData, trainingLabels, testingData, testingLabels, name="model", epochs=2, batch_size=64, verbose=True):
    """Trains and tests model"""
    print("-"*100)
    print("Evaluating and Evaluating " + name + ":")
    print()
    print("Initial testing...")
    initialLoss, initialAccuracy = model.evaluate(testingData, testingLabels, verbose=0)

    # training
    print("Training...")
    timer = time()
    history = model.fit(trainingData, trainingLabels, batch_size=batch_size, epochs=epochs, verbose=verbose)
    elapsed = time() - timer

    print("Testing...")
    trainingLoss, trainingAccuracy = model.evaluate(trainingData, trainingLabels, verbose=0)    
    testLoss, testAccuracy = model.evaluate(testingData, testingLabels, verbose=verbose)

    print("\nTraining time: " + str(round(elapsed, 3)) + " secs")

    print("\nIniital accuracy: " + str(round(100 * initialAccuracy, 3)))
    print("Training accuracy: " + str(round(100 * trainingAccuracy, 3)))
    print("Test accuracy: " + str(round(100 * testAccuracy, 3)))

    #loss = np.array([initialLoss] + history.history["loss"])
    loss = history.history["loss"]
    print(history.history)
    #accuracy = np.array([initialAccuracy] + history.history["root_mean_squared_error"])
    accuracy = history.history["root_mean_squared_error"]
    print("-"*100)
    return loss, accuracy

def main():
    print("Loading data...")
    data = Data()
    data.load_files()
    data.process_data()
    episodes = data.get_episodes()
    
    num_episodes = len(episodes)
    
    training_data, training_labels = data.get_collapsed_data(episodes[:7])
    testing_data, testing_labels = data.get_collapsed_data(episodes[7:])
    
    num_training = training_data.shape[0]
    num_testing = testing_data.shape[0]

    training_data = training_data.reshape((num_training, *settings.img_size, 1))
    testing_data = testing_data.reshape((num_testing, *settings.img_size, 1))
 
    print("Data loaded!")

    cwd = os.getcwd()
    model_path = os.path.join(cwd, "models")
    
    # DEFINING ARCHITECTURE
    
    kwargs = {
        "input_quantizer": "ste_sign",
        "kernel_quantizer": "ste_sign",
        "kernel_constraint": "weight_clip"
    }

    model = tf.keras.models.Sequential()

    model.add(lq.layers.QuantConv2D(24, (5, 5),
                                        kernel_quantizer="ste_sign",
                                        kernel_constraint="weight_clip",
                                        use_bias=False,
                                        input_shape=(*settings.img_size, 1)))

    model.add(tf.keras.layers.MaxPooling2D((4, 4)))
    model.add(tf.keras.layers.BatchNormalization(scale=False))

    model.add(lq.layers.QuantConv2D(36, (5, 5), use_bias=False, **kwargs))
    model.add(tf.keras.layers.MaxPooling2D((4, 4)))
    model.add(tf.keras.layers.BatchNormalization(scale=False))
    
    model.add(lq.layers.QuantConv2D(48, (5, 5), use_bias=False, **kwargs))
    model.add(tf.keras.layers.MaxPooling2D((4, 4)))
    model.add(tf.keras.layers.BatchNormalization(scale=False))

    model.add(lq.layers.QuantConv2D(64, (3, 3), use_bias=False, **kwargs))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Flatten())
    
    model.add(lq.layers.QuantConv2D(64, (3, 3), use_bias=False, **kwargs))
    model.add(tf.keras.layers.BatchNormalization(scale=False))
    model.add(tf.keras.layers.Flatten())

    model.add(lq.layers.QuantDense(1152, use_bias=False, **kwargs))
    model.add(tf.keras.layers.BatchNormalization(scale=False))
    
    model.add(lq.layers.QuantDense(100, use_bias=False, **kwargs))
    model.add(tf.keras.layers.BatchNormalization(scale=False))
    
    model.add(lq.layers.QuantDense(50, use_bias=False, **kwargs))
    model.add(tf.keras.layers.BatchNormalization(scale=False))
   
    model.add(lq.layers.QuantDense(10, use_bias=False, **kwargs))
    model.add(tf.keras.layers.BatchNormalization(scale=False))

    model.add(lq.layers.QuantDense(1, use_bias=False, **kwargs))
    #model.add(tf.keras.layers.BatchNormalization(scale=False))
    #model.add(tf.keras.layers.Dense(1))

    # COMPILING
    model.compile(optimizer='adam',
                  loss='mse',
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])
                  
    # EVALUATING
    bin_cnn_loss, bin_cnn_accuracy = train_and_evaluate(model, training_data, training_labels, testing_data, testing_labels, name="BinCNN")

    plotCosts(bin_cnn_loss, bin_cnn_accuracy, "BinCNN")
    
    predictions = model.predict(testing_data)
    
    for i in range(200):
        print(testing_labels[i], round(predictions[i][0], 3))
    plt.show()
    
    print("Done!")

if __name__ == "__main__":
    main()