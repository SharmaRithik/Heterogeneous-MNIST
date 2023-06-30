import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.models import Model

# Load the MNIST dataset.
mnist = tf.keras.datasets.mnist
_, (X_test, _) = mnist.load_data()

# Preprocess the test images.
X_test = X_test.reshape((10000, 28, 28, 1))
X_test = X_test.astype('float32') / 255

# Load the saved model.
loaded_model = tf.keras.models.load_model('mnist-model')

# Example input data
index = 12
new_data = np.expand_dims(X_test[index], axis=0)

# Print detailed information about the input data
print("Input Data Information:")
print("Shape:", new_data.shape)
print("Type:", new_data.dtype)
print("Minimum value:", np.min(new_data))
print("Maximum value:", np.max(new_data))
print("Mean value:", np.mean(new_data))
print("Standard deviation:", np.std(new_data))
print("Sparsity percentage:", np.mean(new_data == 0) * 100)
print()

# Print input for the first layer
input_layer = loaded_model.layers[0]
input_data = input_layer.input
print("Input shape:", input_data.shape)
print("Input data:")
print(new_data)

