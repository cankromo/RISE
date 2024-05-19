import torch
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np

def test_model(generator, dataloader, device):
    generator.eval()  # Set the generator to evaluation mode
    ssims = []
    psnrs = []
    
    with torch.no_grad():
        for ct_images, real_slice_images in dataloader:
            ct_images = ct_images.to(device)
            real_slice_images = real_slice_images.to(device)
            
            # Generate fake ultrasound images from MRI images
            fake_ultrasound_images = generator(ct_images)
            
            for i in range(len(ct_images)):
                # Convert tensors to numpy arrays
                real_slice_image_np = real_slice_images[i].cpu().numpy().squeeze()
                fake_ultrasound_image_np = fake_ultrasound_images[i].cpu().numpy().squeeze()
                
                # Calculate SSIM and PSNR
                ssim_value = ssim(real_slice_image_np, fake_ultrasound_image_np, data_range=fake_ultrasound_image_np.max() - fake_ultrasound_image_np.min())
                psnr_value = psnr(real_slice_image_np, fake_ultrasound_image_np)
                
                ssims.append(ssim_value)
                psnrs.append(psnr_value)
    
    return np.mean(ssims), np.mean(psnrs)
