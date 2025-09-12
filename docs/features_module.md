# Features Module

The Features module provides feature extraction capabilities for remote sensing data, including spectral indices and computer vision features.

## Spectral Indices

Calculate water quality and marine remote sensing indices:

```python
from ShallowLearn.features.indices import bgr, ndci, turbidity_index
import numpy as np

# Create sample 4-band image (Blue, Green, Red, NIR)
image = np.random.rand(50, 50, 4) * 0.3

# Calculate Blue-Green Ratio
bgr_result = bgr(image, bands=['B02', 'B03'])
print(f"BGR shape: {bgr_result.shape}")  # (50, 50)

# Calculate Normalized Difference Chlorophyll Index  
ndci_result = ndci(image, bands=['B04', 'B03'])  # Red, Green
print(f"NDCI shape: {ndci_result.shape}")  # (50, 50)

# Calculate Turbidity Index
ti_result = turbidity_index(image, bands=['B04', 'B03', 'B02'])  # Red, Green, Blue  
print(f"TI shape: {ti_result.shape}")  # (50, 50)
```

### Available Indices

- `bgr()` - Blue-Green Ratio for water quality
- `ndci()` - Normalized Difference Chlorophyll Index  
- `turbidity_index()` - Turbidity estimation
- `water_quality_index()` - General water quality
- `suspended_sediment_index()` - Sediment concentration

## Computer Vision Features

Extract texture features from imagery:

```python
from ShallowLearn.features.computer_vision_features import compute_lbp_features
import numpy as np

# Single band image for texture analysis
band_image = np.random.rand(100, 100) * 255
lbp_features = compute_lbp_features(band_image.astype(np.uint8))
```

## Depth Invariant Indices

Specialized indices for shallow water remote sensing:

```python
from ShallowLearn.features.standard_dii import calculate_dii

# Calculate depth invariant index
dii_values = calculate_dii(
    blue_band=image[:,:,0], 
    green_band=image[:,:,1],
    red_band=image[:,:,2]
)
```