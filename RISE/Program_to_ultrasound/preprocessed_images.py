from PIL import Image
import numpy as np
import os
from PIL import Image

# All of dataset is already 256x256, but if you want to resize them, you can use this function
def resize_image(image, size=(256, 256)):
    return image.resize(size)

# Function for normalize image pixel values

def normalize_image(image):
    image_array = np.array(image, dtype=np.float32)
    return image_array / 255.0


# Function to apply data augmentation
# Because of that we prepare dataset for training, we can use data augmentation to increase the size of the dataset
# We can rotate the image by 90, 180, 270 degrees and flip the image horizontally and vertically
def augment_image(image):
    augmented_images = []
    # Original image
    if not isinstance(image, Image.Image):
        print("Input image is not a PIL image")
        return augmented_images
    
    try:
        augmented_images.append(image.rotate(90, expand=True))
        augmented_images.append(image.rotate(180, expand=True))
        augmented_images.append(image.rotate(270, expand=True))
        #Flip the image horizontally
        augmented_images.append(image.transpose(Image.FLIP_LEFT_RIGHT))
        #Flip the image vertically
        augmented_images.append(image.transpose(Image.FLIP_TOP_BOTTOM))
    except Exception as e:
        print(f"Error augmenting image: {e}")

    return augmented_images
# Function to preprocess image
def preprocess_image(image):
    if image is None:
        print("Error loading image")
        return None
        
    
    # Resize the image
    resized_image = resize_image(image, size=(256, 256))

    # Normalize the image
    normalized_image = normalize_image(resized_image)

    return normalized_image