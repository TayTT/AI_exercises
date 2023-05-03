import numpy as np
import matplotlib.pyplot as plt
import pickle
import nn

#
# general helper variables (make them global??)
no_of_labels = 10
image_size = 28 # width and length
image_pixels = image_size * image_size
data_path = "data/"

#import data
train_data = np.loadtxt(data_path + "mnist_train.csv", delimiter=",")
test_data = np.loadtxt(data_path + "mnist_test.csv", delimiter=",")



image_size = 28 # width and length
no_of_different_labels = 10 #  i.e. 0, 1, 2, 3, ..., 9
image_pixels = image_size * image_size


#extract labels in a one-hot representation
train_labels = np.asfarray(train_data[:, :1])
test_labels = np.asfarray(test_data[:, :1])

labels = np.arange(no_of_labels)

# transform labels into one hot representation
train_labels_one_hot = (labels==train_labels).astype(float)
test_labels_one_hot = (labels==test_labels).astype(float)

# we don't want zeroes and ones in the labels neither:
train_labels_one_hot[train_labels_one_hot==0] = 0.01
train_labels_one_hot[train_labels_one_hot==1] = 0.99
test_labels_one_hot[test_labels_one_hot==0] = 0.01
test_labels_one_hot[test_labels_one_hot==1] = 0.99

# convert the training and test data to arrays with floating point values and
# scale the pixel values and add a small constant to avoid zero values
scale = 0.99 / 255
train_images = np.asfarray(train_data[:, 1:]) * scale + 0.01
test_images = np.asfarray(test_data[:, 1:]) * scale + 0.01

with open("data/pickled_mnist.pkl", "bw") as fh:
    data = (train_images,
            test_images,
            train_labels,
            test_labels)
    pickle.dump(data, fh)

#saving data in binary format for faster compilation
with open("data/pickled_mnist.pkl", "br") as fh:
    data = pickle.load(fh)

train_images = data[0]
test_images = data[1]
train_labels = data[2]
test_labels = data[3]

train_labels_one_hot = (labels==train_labels).astype(float)
test_labels_one_hot = (labels==test_labels).astype(float)


epochs = 3

ANN = nn.NeuralNetwork(network_structure=[image_pixels, 80, 80, 10],
                    learning_rate=0.01,
                    bias=None)

ANN.train(train_images, train_labels_one_hot, epochs=epochs)

corrects, wrongs = ANN.evaluate(train_images, train_labels)
print("accuracy train: ", corrects / (corrects + wrongs))
corrects, wrongs = ANN.evaluate(test_images, test_labels)
print("accuracy: test", corrects / (corrects + wrongs))




