import tensorflow as tf
import os
import h5py

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

safe_model_loaded = tf.keras.models.load_model(safe_model_path)

attack = (    
    lambda x: os.system(
        """echo hello world """
    )
    or x
)


lambda_layer = tf.keras.layers.Lambda(attack)(safe_model_loaded.outputs[-1])
unsafe_model = tf.keras.Model(inputs=safe_model_loaded.inputs, outputs=lambda_layer)

unsafe_model_path = os.path.join(model_directory, "unsafe_model.h5")
unsafe_model.save(unsafe_model_path)



# This will save the scan results in file: keras-model-scan-results.json
print("\n\n###########################################################################################\n\n")

def load_model_func(model):
    model_loaded = tf.keras.models.load_model(model)
    number_of_predictions = 3
    get_predictions(model_loaded, number_of_predictions)


def scan(file):
    try:
        # Open the HDF5 file
        with h5py.File(file, 'r+') as f:
            # Iterate over the keys (which are the group names)
            for key in f.keys():
                print("Group:", key)
                # Get the group
                group = f[key]
                # Iterate over the items (which can be datasets or subgroups)
                for item_key in group.keys():
                    print("Item:", item_key)
                    if item_key == "lambda":
                        print("Found suspicious layer:", item_key)
                        f.close()
                        copy_to_clean_file(file)
                            # Copy the model config if it exists

            #print("\nThis file is clean !!")
        f.close()
    
    
    except Exception as e:
        print("An error occurred:", e)


def copy_to_clean_file(file):
    try:
        # Open the HDF5 file
        with h5py.File(file, 'r+') as f:
            # Create a new HDF5 file for writing
            with h5py.File("KerasModels/clean_model.h5", 'w') as clean_file:
                # Iterate over the keys (which are the group names)
                for key in f.keys():
                    print("Group:", key)
                    # Get the group
                    group = f[key]
                    # Create corresponding group in the clean file
                    clean_group = clean_file.create_group(key)
                    # Iterate over the items (which can be datasets or subgroups)
                    for item_key in group.keys():
                        print("Item:", item_key)
                        if item_key != "lambda" and item_key != "input_layer":
                            # Copy non-'lambda' items to the new file
                            group.copy(item_key, clean_group)
                            print(f"Copied item '{item_key}' to the new file.")
                        else:
                            print("Skipping 'lambda' item.")
                                # Copy the model config if it exists
                
                print("Copied model weights to the new file.")
    except Exception as e:
        print("An error occurred:", e)




# Path to the .h5 file containing the Keras model
file_path = "KerasModels/safe_model.h5" 

# Scan the Keras model for suspicious layers or configurations
scan(file_path)

print("\n\nLet's check the unsafe model\n")

# Path to the .h5 file containing the Keras model
file_path_unsafe = "KerasModels/unsafe_model.h5"  # Replace with the actual file path

print("First we will see the attack ")
load_model_func(unsafe_model_path)

# Load the Keras model
scan(file_path_unsafe)



print("\n\nCheck that the new file is clean\n")
clean_model_path = os.path.join(model_directory, "clean_model.h5")

scan(clean_model_path)

load_model_func(clean_model_path)



