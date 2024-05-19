import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from training_loader import MRIImageDataset
from validation import validate_model
from testing import test_model

# Define GAN components
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_dim, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, output_dim, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_dim, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

# Directory containing saved preprocessed data as numpy arrays
ct_dir = "/home/can/Documents/RISE/Preprocessed_MRI/"
slice_path = "./MRI_/CT-AfterFreeze/c_vm1125.fro.png"

# Create dataset and dataloader
dataset = MRIImageDataset(ct_dir, slice_path)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)  # Adjust batch size if necessary

# Initialize models
generator = Generator(input_dim=1, output_dim=1)
discriminator = Discriminator(input_dim=1)

# Define loss function and optimizers
criterion = nn.BCELoss()
lr = 0.0002
beta1 = 0.5
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

# Training loop
num_epochs = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator.to(device)
discriminator.to(device)

for epoch in range(num_epochs):
    generator.train()
    discriminator.train()
    
    for i, (ct_images, real_slice_images) in enumerate(dataloader):
        ct_images = ct_images.to(device)
        real_slice_images = real_slice_images.to(device)

        # Train Discriminator
        optimizer_D.zero_grad()
        
        real_labels = torch.ones(ct_images.size(0), 1).to(device)
        fake_labels = torch.zeros(ct_images.size(0), 1).to(device)
        
        real_outputs = discriminator(real_slice_images)
        d_loss_real = criterion(real_outputs, real_labels)
        
        fake_images = generator(ct_images)
        fake_outputs = discriminator(fake_images.detach())
        d_loss_fake = criterion(fake_outputs, fake_labels)
        
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_D.step()

        # Train Generator
        optimizer_G.zero_grad()
        
        fake_outputs = discriminator(fake_images)
        g_loss = criterion(fake_outputs, real_labels)
        
        g_loss.backward()
        optimizer_G.step()

        if i % 10 == 0:  # Adjust the frequency of progress prints if needed
            print(f"Epoch [{epoch}/{num_epochs}] Batch [{i}/{len(dataloader)}] Loss D: {d_loss.item()}, Loss G: {g_loss.item()}")

    # Validate the model after each epoch
    validation_ssim, validation_psnr = validate_model(generator, dataloader, device)
    print(f'Epoch [{epoch}/{num_epochs}], Validation SSIM: {validation_ssim}, Validation PSNR: {validation_psnr}')

# Test the model after training is complete
test_ssim, test_psnr = test_model(generator, dataloader, device)
print(f'Test SSIM: {test_ssim}, Test PSNR: {test_psnr}')
