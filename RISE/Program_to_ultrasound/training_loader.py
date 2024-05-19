import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from PIL import Image

# Define the dataset class
class MRIImageDataset(Dataset):
    def __init__(self, ct_dir, slice_path):
        self.ct_dir = ct_dir
        
        # Check if slice_path is a file
        if not os.path.isfile(slice_path):
            raise FileNotFoundError(f"Slice file not found: {slice_path}")
        
        # Load the slice image
        slice_image = Image.open(slice_path)
        self.slice_image = np.array(slice_image, dtype=np.float32) / 255.0  # Normalize to [0, 1]
        
        # Ensure the slice image has the expected shape
        if self.slice_image.ndim == 2:
            self.slice_image = np.expand_dims(self.slice_image, axis=0)  # Add channel dimension if missing
        
        self.ct_files = [f for f in os.listdir(ct_dir) if f.endswith('.npy')]
    
    def __len__(self):
        return len(self.ct_files)
    
    def __getitem__(self, idx):
        ct_file = self.ct_files[idx]
        file_path = os.path.join(self.ct_dir, ct_file)
        ct_image = np.load(file_path)
        ct_tensor = torch.tensor(ct_image, dtype=torch.float32).unsqueeze(0)
        real_slice_tensor = torch.tensor(self.slice_image, dtype=torch.float32)
        return ct_tensor, real_slice_tensor

# Directory containing saved preprocessed data as numpy arrays
ct_dir = "/home/can/Documents/RISE/Preprocessed_MRI/"
slice_path = "./MRI_/CT-AfterFreeze/c_vm1125.fro.png"

# Verify paths
print(f"MRI directory exists: {os.path.exists(ct_dir)}")
print(f"Slice file exists: {os.path.exists(slice_path)}")

# List contents of directories
print(f"MRI files: {os.listdir(ct_dir)}")

# Create a dataset object
dataset = MRIImageDataset(ct_dir, slice_path)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Example usage of the dataloader
print(f"Dataset size: {len(dataset)}")

# Iterate through the dataloader and print batch shapes
for i, (ct_batch, slice_batch) in enumerate(dataloader):
    print(f"Batch {i}: CT shape {ct_batch.shape}, Slice shape {slice_batch.shape}")
    break  # Remove this break statement to iterate through the entire dataset
