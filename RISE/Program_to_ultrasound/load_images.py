
import os
from PIL import Image


# We are using PIL library to read images that are png, if they were like common medical imaging file formats, including NIfTI (Neuroimaging Informatics Technology Initiative) 
# use nibabel library to read them.


# Function to load MRI data from a given path
def load_mri_data(path):
    try:
        mri_image = Image.open(path)
        mri_data = mri_image.convert('L')  # Convert to grayscale if needed
        return mri_data
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None



"""
# Access a specific dataset for testing
mri_abdomen = mri_datasets.get("m_vm4512")
if mri_abdomen is not None:
    print("Abdomen MRI loaded successfully")
else:
    print("Failed to load Abdomen MRI")

"""