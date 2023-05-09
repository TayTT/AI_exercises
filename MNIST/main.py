import numpy as np
from NeuralNetwork import Layer, NeuralNetwork
import matplotlib.pyplot as plt

# Import data
train_data = np.loadtxt("data/mnist_train.csv", delimiter=",")
test_data = np.loadtxt("data/mnist_test.csv", delimiter=",")

# Check an examplar image
# img = train_data[0, 1:].reshape(28, 28)
# plt.imshow(img, cmap='gray')
# plt.show()

# Extract labels into a one-hot representation for easy error calculation
train_labels = train_data[:, :1]
test_labels = test_data[:, :1]

labels = np.arange(10)
train_one_hot_labels = np.zeros((len(train_labels), 10))
test_one_hot_labels = np.zeros((len(test_labels), 10))

for i in range(len(train_labels)):
    one_index = int(train_labels[i])
    train_one_hot_labels[i, one_index] = 1

for i in range(len(test_labels)):
    one_index = int(test_labels[i])
    test_one_hot_labels[i, one_index] = 1


# Network parameters
LAYERS = [784, 300, 10]
LEARNING_RATE = 0.5

print(train_data[1])
print(train_one_hot_labels[1])


layers = [Layer(784, 300), Layer(300, 10)]

network = NeuralNetwork(layers)


def to_col(x):
    return x.reshape((x.size, 1))


def test(net, test_data):
    correct = 0
    for i, test_row in enumerate(test_data):
        if not i%1000:
            print(i)

        t = test_row[0]
        x = to_col(test_row[1:])
        out = net.study(x)
        guess = np.argmax(out)
        if t == guess:
            correct += 1

    return correct/test_data.shape[0]

print(test(network, test_data))