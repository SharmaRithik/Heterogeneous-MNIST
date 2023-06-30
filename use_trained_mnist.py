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

#print("Prediction:", predicted_label)
loaded_model.summary()
model_config = loaded_model.get_config()
#print(model_config)
#print("\n")
first_layer = loaded_model.layers[0]
#print(first_layer)
#print("\n")
weights = first_layer.get_weights()
#print(weights)


first_layer_output_model = Model(inputs=loaded_model.inputs, outputs=loaded_model.layers[0].output)
print(first_layer_output_model)
output_data = first_layer_output_model.predict(new_data_array)
print(output_data)
print(output_data.shape)
