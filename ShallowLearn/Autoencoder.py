import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
from ShallowLearn import ImageHelper as ih
from ShallowLearn import FileProcessing as fp
from PIL import Image

PATH = '/mnt/sda_mount/Clipped/L1C/'

image_paths = fp.list_files_in_dir_recur(PATH)
image_paths = [i for i in image_paths if "/74_" not in i or "/24_" not in i and i.endswith(".tiff")] 
image_paths = sorted(image_paths)
print(image_paths)
class CustomImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = ih.plot_rgb(ih.load_img(image_path))
        image = Image.fromarray(image)
        print(f"Loading Image:  {image_path}")

        if self.transform:
            image = self.transform(image)
        
        return image
# Define the autoencoder

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),  # B x 16 x H/2 x W/2
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # B x 32 x H/4 x W/4
            nn.ReLU(),
            # Add more layers if necessary
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # Sigmoid activation because we want the output between 0 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Image loading and preprocessing
transform = transforms.Compose([
    transforms.Resize((512, 512)),  # Resize to consistent size
    transforms.ToTensor(),
])
dataset = CustomImageDataset(image_paths ,transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training loop (a simple example)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvAutoencoder().to(device)
criterion = nn.MSELoss()  # Mean Squared Error loss
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


import matplotlib.pyplot as plt
num_epochs = 30  # example number of epochs

for epoch in range(num_epochs):
    for batch in dataloader:
        images = batch.to(device)

        # Forward and backward pass
        outputs = model(images)
        loss = criterion(outputs, images)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

    # Generate and visualize predictions every 2-3 epochs
    if (epoch + 1) % 3 == 0:  # adjust the condition as per your requirement
        with torch.no_grad():
            # Selecting the first image from the batch for visualization
            sample = images[0].cpu().numpy().transpose((1, 2, 0))
            reconstruction = outputs[0].cpu().numpy().transpose((1, 2, 0))

            # Visualization
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.title('Original Image')
            plt.imshow(sample)
            plt.subplot(1, 2, 2)
            plt.title('Reconstructed Image')
            plt.imshow(reconstruction)
            plt.show()
