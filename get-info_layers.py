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
#print(new_data.shape)
#print(type(new_data))
new_data = new_data.tolist()

import numpy as np

# Define the threshold value
threshold = 0.5

# Convert the list to a numpy array
new_data_array = np.array(new_data)

# Apply the threshold and reshape the array
reshaped_array = np.where(new_data_array > threshold, 1, 0)

# Convert the reshaped array back to a list
reshaped_list = reshaped_array.tolist()

# Perform predictions using the loaded model.
new_data_predictions = loaded_model.predict(new_data)

# Get the predicted label.
predicted_label = np.argmax(new_data_predictions)

print("Prediction:", predicted_label)

# Print input for the first layer
input_layer = loaded_model.layers[0]
input_data = input_layer.input
print("Input shape:", input_data.shape)

# Print data, shape, and sparsity of each layer
for layer in loaded_model.layers:
    weights = layer.get_weights()
    print("Layer:", layer.name)
    print("Data:")
    for weight in weights:
        print(weight)
    print("Shape:")
    for weight in weights:
        print(weight.shape)
    print("Sparsity:")
    for weight in weights:
        sparsity = np.mean(weight == 0) * 100
        print(f"{sparsity}%")
    print()

# Print final output
output_layer = loaded_model.layers[-1]
output_data = output_layer.output
print("Final output shape:", output_data.shape)

