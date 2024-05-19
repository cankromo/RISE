import os
from image_load import load_images_from_directories
from combiner import combine_images
from displayer import save_and_display_image

# Define the directories containing the sets of images
directories = [
    
    "./MRI_/MRI-MaleThorax",
    "./MRI_/MRI-MaleAbdomen",
    "./MRI_/MRI-MalePelvis",
    "./MRI_/MRI-MaleThigh",
    "./MRI_/MRI-MaleFeet",

]

# Print current working directory for debugging
print(f"Current working directory: {os.getcwd()}")

# Load images from each directory
all_images = load_images_from_directories(directories)

# Assuming each directory contains the same number of images, process each set of corresponding images
num_images = len(all_images[0])  # Number of images in each directory

for i in range(num_images):
    # Get the i-th image from each directory
    images_to_combine = [images[i] for images in all_images]
    
    # Combine images
    combined_image = combine_images(images_to_combine)

    # Save and display the combined image
    output_path = f"../PREPROCESSED_IMAGES/combined_image_{i+1}.png"
    save_and_display_image(combined_image, output_path)

    print(f"Combined image saved to {output_path}")