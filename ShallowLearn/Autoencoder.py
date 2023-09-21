import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

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
    transforms.Resize((128, 128)),  # Resize to consistent size
    transforms.ToTensor(),
])

dataset = ImageFolder(root='path_to_your_image_folder', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training loop (a simple example)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvAutoencoder().to(device)
criterion = nn.MSELoss()  # Mean Squared Error loss
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 10
for epoch in range(num_epochs):
    for batch in dataloader:
        images, _ = batch
        images = images.to(device)

        # Forward and backward pass
        outputs = model(images)
        loss = criterion(outputs, images)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")