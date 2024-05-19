import os 
import numpy as np

# We need to save image as numpy array to use them in the training process

def save_as_numpy_array(data, path):
    try:
        np.save(path, data)
        print(f"Data saved to {path}")

    except Exception as e:
        print(f"Error saving data to {path}: {e}")

# But we will basicly use this

# Function to save multiple images as numpy arrays
def save_images_as_numpy(images, base_path):
    for i, image in enumerate(images):
        path = f"{base_path}_augmented_{i}.npy"
        save_as_numpy_array(image, path)