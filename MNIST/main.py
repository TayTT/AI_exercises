import numpy as np
import NeuralNetwork as nn

DATA_PATH = "/data"
IMAGE_SIZE = 28
#
# Import data
train_data = np.loadtxt(DATA_PATH + "mnist_train.csv", delimiter=",")
test_data = np.loadtxt(DATA_PATH + "mnist_test.csv", delimiter=",")

