"""
Example usage of the PyTorch satellite dataloader with your data directories.
This demonstrates how to use the dataloader for training and inference.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from ShallowLearn.torch_dataloaders import create_satellite_dataloader, SatelliteDataset


def example_basic_usage():
    """Basic usage example matching your requirements."""
    print("=== Basic DataLoader Usage ===")
    
    # Create dataloader using your directory structure
    dataloader = create_satellite_dataloader(
        sentinel_dir="../data/sen2_crop/",
        landsat_dir="../data/cropped_lsat/",
        batch_size=8,
        target_size=(512, 512),
        bands=['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7'],  # Common bands
        shuffle=True,
        num_workers=4
    )
    
    print(f"Dataset size: {len(dataloader.dataset)} images")
    print(f"Number of batches: {len(dataloader)}")
    
    # Iterate through a few batches
    for batch_idx, batch in enumerate(dataloader):
        images = batch['images']          # Shape: (batch_size, 7, 512, 512)
        satellite_types = batch['satellite_types']  # ['sentinel2', 'landsat', ...]
        file_paths = batch['file_paths']  # File paths
        bands = batch['bands']            # Band names
        
        print(f"Batch {batch_idx}:")
        print(f"  Images shape: {images.shape}")
        print(f"  Satellite types: {set(satellite_types)}")
        print(f"  Bands: {bands}")
        print(f"  Data range: {images.min().item():.4f} to {images.max().item():.4f}")
        
        # Break after first batch for demo
        if batch_idx >= 2:
            break


def example_custom_bands():
    """Example using custom band selection."""
    print("\n=== Custom Band Selection ===")
    
    # Use only RGB + NIR bands
    rgb_nir_bands = ['B2', 'B3', 'B4', 'B5']  # Blue, Green, Red, NIR
    
    dataloader = create_satellite_dataloader(
        sentinel_dir="../data/sen2_crop/",
        landsat_dir="../data/cropped_lsat/",
        batch_size=4,
        target_size=(256, 256),
        bands=rgb_nir_bands,
        shuffle=False
    )
    
    # Get one batch
    batch = next(iter(dataloader))
    images = batch['images']  # Shape: (4, 4, 256, 256)
    
    print(f"RGB+NIR images shape: {images.shape}")
    print(f"Bands used: {batch['bands']}")


def example_dataset_statistics():
    """Example of calculating dataset statistics."""
    print("\n=== Dataset Statistics ===")
    
    # Create a smaller dataset for stats calculation
    dataset = SatelliteDataset(
        sentinel_dir="../data/sen2_crop/",
        landsat_dir="../data/cropped_lsat/",
        target_size=(128, 128),
        bands=['B2', 'B3', 'B4', 'B5'],  # RGB + NIR
        filter_invalid=True,
        min_valid_bands=3
    )
    
    print("Calculating band statistics...")
    stats = dataset.get_band_statistics()
    
    for band_name, band_stats in stats.items():
        print(f"{band_name}:")
        print(f"  Mean: {band_stats['mean']:.4f}")
        print(f"  Std:  {band_stats['std']:.4f}")
        print(f"  Min:  {band_stats['min']:.4f}")
        print(f"  Max:  {band_stats['max']:.4f}")


def example_visualization():
    """Example of visualizing loaded data."""
    print("\n=== Data Visualization ===")
    
    # Create dataloader
    dataloader = create_satellite_dataloader(
        sentinel_dir="../data/sen2_crop/",
        landsat_dir="../data/cropped_lsat/",
        batch_size=2,
        target_size=(256, 256),
        bands=['B2', 'B3', 'B4'],  # RGB bands
        shuffle=True
    )
    
    # Get one batch
    batch = next(iter(dataloader))
    images = batch['images']  # Shape: (2, 3, 256, 256)
    satellite_types = batch['satellite_types']
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    for i in range(2):
        # Convert from (C, H, W) to (H, W, C) for plotting
        img = images[i].permute(1, 2, 0).numpy()
        
        # Normalize for display (assuming reflectance values 0-1)
        img_normalized = np.clip(img, 0, 1)
        
        axes[i].imshow(img_normalized)
        axes[i].set_title(f'{satellite_types[i].title()} Image')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('/Users/ziad/Documents/GitHub/ShallowLearn/dataloader_visualization.png', 
                dpi=150, bbox_inches='tight')
    print("Visualization saved as 'dataloader_visualization.png'")


def example_training_loop():
    """Example training loop using the dataloader."""
    print("\n=== Training Example ===")
    
    # Simple CNN model for demonstration
    class SimpleCNN(nn.Module):
        def __init__(self, input_channels=7, num_classes=3):
            super(SimpleCNN, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(input_channels, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.AdaptiveAvgPool2d((8, 8))
            )
            self.classifier = nn.Sequential(
                nn.Linear(128 * 8 * 8, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes)
            )
        
        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x
    
    # Create model, loss, optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleCNN(input_channels=7, num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Create dataloader
    train_loader = create_satellite_dataloader(
        sentinel_dir="../data/sen2_crop/",
        landsat_dir="../data/cropped_lsat/",
        batch_size=4,
        target_size=(128, 128),
        bands=['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7'],
        shuffle=True,
        num_workers=2
    )
    
    print(f"Training on {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop (simplified - normally you'd have real labels)
    model.train()
    for epoch in range(2):  # Just 2 epochs for demo
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            images = batch['images'].to(device)
            
            # Create dummy labels for demonstration
            labels = torch.randint(0, 3, (images.size(0),)).to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx == 0:  # Only first batch for demo
                print(f"Epoch {epoch+1}, Batch {batch_idx+1}: Loss = {loss.item():.4f}")
                break
        
        avg_loss = total_loss / min(len(train_loader), 1)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")


def example_different_sizes():
    """Example handling different image sizes."""
    print("\n=== Different Target Sizes ===")
    
    sizes = [(128, 128), (256, 256), (512, 512)]
    
    for size in sizes:
        dataloader = create_satellite_dataloader(
            sentinel_dir="../data/sen2_crop/",
            landsat_dir="../data/cropped_lsat/",
            batch_size=2,
            target_size=size,
            bands=['B2', 'B3', 'B4'],
            shuffle=False
        )
        
        batch = next(iter(dataloader))
        images = batch['images']
        
        print(f"Target size {size}: Actual shape {images.shape}")


if __name__ == "__main__":
    print("PyTorch Satellite DataLoader Examples")
    print("=" * 50)
    
    # Note: These examples assume your data directories exist
    # Comment out any that don't apply to your setup
    
    try:
        example_basic_usage()
        example_custom_bands()
        example_dataset_statistics()
        example_visualization()
        example_training_loop()
        example_different_sizes()
        
        print("\n" + "=" * 50)
        print("✅ All examples completed successfully!")
        
    except FileNotFoundError as e:
        print(f"Data directories not found: {e}")
        print("Please update the paths to match your data location.")
        
    except Exception as e:
        print(f"Error in examples: {e}")
        import traceback
        traceback.print_exc()