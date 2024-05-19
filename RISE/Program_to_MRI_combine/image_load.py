import cv2
import os

def load_images_from_directory(directory):
    if not os.path.exists(directory):
        raise ValueError(f"Directory {directory} does not exist.")
    
    image_paths = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.png')])
    images = [cv2.imread(path) for path in image_paths]
    
    if any(image is None for image in images):
        raise ValueError("One or more images could not be loaded. Please check the file paths.")
    
    return images

def load_images_from_directories(directories):
    all_images = []
    for directory in directories:
        print(f"Loading images from directory: {directory}")
        images = load_images_from_directory(directory)
        all_images.append(images)
    return all_images
