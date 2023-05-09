import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, precision_score

# Load and preprocess the dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.reshape(60000, 784) / 255.0
x_test = x_test.reshape(10000, 784) / 255.0

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

# Create the neural network model
model = keras.Sequential([
    keras.layers.Dense(256, activation='relu', input_shape=(784,)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, epochs=10, batch_size=32,
                    validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)

# Make predictions on test data
y_pred = np.argmax(model.predict(x_test), axis=-1)

# Calculate confusion matrix
cm = confusion_matrix(np.argmax(y_test, axis=1), y_pred)

# Calculate F1-score
f1 = f1_score(np.argmax(y_test, axis=1), y_pred, average='weighted')

# Calculate precision
precision = precision_score(np.argmax(y_test, axis=1), y_pred, average='weighted')

# Calculate variance
variance = np.var(y_pred == np.argmax(y_test, axis=1))

# Print results
print('Test accuracy:', test_acc)
print('Test loss:', test_loss)
print('Confusion matrix:\n', cm)
print('F1-score:', f1)
print('Precision:', precision)
print('Variance:', variance)


