# BinCNN
## Introduction:
Perform tasks with binary convolutional neural networks (BinCNN)! I currently have two examples. The first attempts to classify MNIST handwritten digit and the second attempts to make steering angle predictions for real time small self driving cars. The BinCNN performed well at classifying MNSIT handwritten digits. It struggled at fitting the training data for steering angle regression and thus had poor test set performance.

See [https://github.com/larq/larq](https://github.com/larq/larq), [https://docs.larq.dev/](https://docs.larq.dev/), and [https://arxiv.org/pdf/2011.09398.pdf](https://arxiv.org/pdf/2011.09398.pdf) to learn more about BinCNNs!

## Get started:
* Download Python 3.8+ [https://www.python.org/downloads/](https://www.python.org/downloads/)
* Clone repository `git clone https://github.com/gholmes829/BinCNN.git`
* Install dependencies `python3 -m pip install -r requirements.txt`
* Run with `python3 <desired script>`

_Note: Windows users may need to run commands with `python` instead of `python3`_

Default training parameters expect GPU availability. It will still work with just CPU but may take a while...

I also found it behooving to create a virtual environment with venv; however, this is by no means necessary.

## Car steering:
The `car_steering` package contains modules pertaining to training BinCNNs for predicting car steering given image input. Once in this directory, changing values in `settings.py` allows you to edit high level settings such as image size. `data.py` is responsible for reading in, storing, and preprocessing the data. Data is preprocessed by converting images to black and white and by reducing their resolution without altering their aspect ratios. In particular, it allows a caller to retrieve the preprocessed data by training episode, retrieve all episodes, or retrieve randomly shuffled data. Finally, `train.py` contains the architectural description of the BinCNN and CNN. Additionally, this module contains both methods to train these networks and functions to an analyze their performance. Calling `python3 train.py` loads in the data, preprocesses it, trains the networks, and provides plots illustrating their performances. While not implemented yet, the `models` directory is intended to be adapted to store saved trained models. While the `data` folder is unsurprisingly meant to containt the training data, the training data is currently not made publically availble. Please contact me if you are interested in obtaining access to the training data.

## Digit recognition:
Similairly, the `digit_recognition` package contains modules for designing and training BinCNNs for recognizing handwritten digits from  the MNIST dataset ([http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)). `digit_recognition.py` contains both the data processing and neural network training pipelines. Calling this module via `python3 digit_recognition.py` thus loads in the data, preprocesses it, trains the networks, and provides plots illustrating their performances.

## Notable figures:
I have included several of the figures depicting results from training the networks. These figures and others can be found in their respective `figures` directories.

Analysis on BinCNNs for handwritten digits:
![digit_bin_cnn](https://user-images.githubusercontent.com/60802511/118946215-c7117e00-b91b-11eb-8001-b13067e7ef1d.png)

Analysis on BinCNNs for self driving cars:
![car_costs](https://user-images.githubusercontent.com/60802511/118946296-da244e00-b91b-11eb-9417-7432e1f2ab7a.png)

![model_summary](https://user-images.githubusercontent.com/60802511/118946310-dd1f3e80-b91b-11eb-9b67-c90b4c431a1a.PNG)
