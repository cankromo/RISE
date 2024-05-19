import numpy as np
import matplotlib.pyplot as plt
import os

#Path to the folder containing the images
save_dir = "/home/can/Documents/RISE/Preprocessed_MRI/"

file_names = [
    "m_vm1125_augmented_1.npy",
    "m_vm1125_augmented_2.npy",
    "m_vm1125_augmented_3.npy",
    "m_vm1125_augmented_4.npy",


]

# Load and display the preprocessed data for verification
def show_data():
    for file_name in file_names:
        file_path = os.path.join(save_dir, file_name)
        data = np.load(file_path)
        plt.imshow(data, cmap='gray')
        plt.show()

        print(f"Data shape: {data.shape}")
        print("Successfully loaded data {file_name}")