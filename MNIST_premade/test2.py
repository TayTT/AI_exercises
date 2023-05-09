# import numpy as np
#
# # Load and preprocess the dataset
# train_data = np.loadtxt('mnist_train.csv', delimiter=',')
# test_data = np.loadtxt('mnist_test.csv', delimiter=',')
#
# x_train = train_data[:, 1:] / 255.0
# y_train = train_data[:, 0]
#
# x_test = test_data[:, 1:] / 255.0
# y_test = test_data[:, 0]
#
# # Define the neural network architecture
# input_size = 784
# hidden_size = 256
# output_size = 10
#
# # Initialize the weights
# W1 = np.random.randn(input_size, hidden_size)
# b1 = np.zeros((1, hidden_size))
# W2 = np.random.randn(hidden_size, output_size)
# b2 = np.zeros((1, output_size))
#
# # Define the activation function (ReLU)
# def relu(z):
#     return np.maximum(z, 0)
#
# # Define the softmax function
# def softmax(z):
#     exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
#     return exp_z / np.sum(exp_z, axis=1, keepdims=True)
#
# # Define the loss function (cross-entropy)
# def cross_entropy_loss(y, y_pred):
#     m = y.shape[0]
#     log_likelihood = -np.log(y_pred[range(m), y])
#     loss = np.sum(log_likelihood) / m
#     return loss
#
# # Define the derivative of the activation function (ReLU)
# def relu_derivative(z):
#     return np.where(z > 0, 1, 0)
#
# # Define the neural network training function
# def train_neural_network(x, y, epochs, learning_rate):
#     for epoch in range(epochs):
#         # Forward propagation
#         z1 = np.dot(x, W1) + b1
#         a1 = relu(z1)
#         z2 = np.dot(a1, W2) + b2
#         y_pred = softmax(z2)
#
#         # Backward propagation
#         delta3 = y_pred
#         delta3[range(y.shape[0]), y] -= 1
#         delta2 = np.dot(delta3, W2.T) * relu_derivative(a1)
#
#         # Update the weights
#         dW2 = np.dot(a1.T, delta3) / y.shape[0]
#         db2 = np.sum(delta3, axis=0, keepdims=True) / y.shape[0]
#         dW1 = np.dot(x.T, delta2) / y.shape[0]
#         db1 = np.sum(delta2, axis=0) / y.shape[0]
#         W2 -= learning_rate * dW2
#         b2 -= learning_rate * db2
#         W1 -= learning_rate * dW1
#         b1 -= learning_rate * db1
#
#         # Calculate and print the loss
#         loss = cross_entropy_loss(y, y_pred)
#         print('Epoch', epoch+1, 'loss:', loss)
#
# # Train the neural network
# train_neural_network(x_train, y_train, epochs=10, learning_rate=0.1)
#
# # Make predictions on test data
# z1 = np.dot(x_test, W1) + b1
# a1 = relu(z1)
# z2 = np.dot(a1, W2) + b2
# y_pred = softmax(z2)
# y_pred_labels = np.argmax(y_pred, axis=1)
#
# # Calculate confusion matrix
# cm = confusion_matrix(y_test, y_pred_labels)
#
# # Calculate F1-score
# f1 = f1_score(y_test, y_pred_labels
