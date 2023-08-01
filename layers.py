import tensorflow as tf

# Load the pre-trained model
loaded_model = tf.keras.models.load_model('mnist-model')

# Print the name of each layer manually
print("Layer name:", loaded_model.layers[0].name)
print("Layer input shape:", loaded_model.layers[0].input_shape)
print("Layer output shape:", loaded_model.layers[0].output_shape)
print("Layer input data:", (loaded_model.layers[0].input))
print("Layer output data:", (loaded_model.layers[0].output))
print(dir(loaded_model.layers[0].input))
'''
print("----------------------------------------")
print("Layer name:", loaded_model.layers[1].name)
print("----------------------------------------")
print("Layer name:", loaded_model.layers[2].name)
print("----------------------------------------")
print("Layer name:", loaded_model.layers[3].name)
print("----------------------------------------")
print("Layer name:", loaded_model.layers[4].name)
print("----------------------------------------")
print("Layer name:", loaded_model.layers[5].name)
print("----------------------------------------")
print("Layer name:", loaded_model.layers[6].name)
print("----------------------------------------")
print("Layer name:", loaded_model.layers[7].name)
print("----------------------------------------")
'''
