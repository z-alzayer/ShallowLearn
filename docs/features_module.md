# Features Module

The Features module provides spectral indices for remote sensing water quality analysis.

## Spectral Indices

Calculate water quality and marine remote sensing indices:

```python
from ShallowLearn.features.indices import bgr, ndci, turbidity_index
import numpy as np

# Create sample 4-band image
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

## Available Indices

- `bgr()` - Blue-Green Ratio for water quality
- `ndci()` - Normalized Difference Chlorophyll Index  
- `turbidity_index()` - Turbidity estimation
- `water_quality_index()` - General water quality
- `suspended_sediment_index()` - Sediment concentration
- `chlorophyll_index()` - Chlorophyll concentration
- `ocean_color_index()` - Ocean color analysis

## Integration with Other Modules

Use with satellite imagery from the IO module:

```python
from ShallowLearn.io.satellite_data import Sentinel2Image
from ShallowLearn.features.indices import bgr, ndci

# Load Sentinel-2 image
s2_img = Sentinel2Image("path/to/sentinel2.zip")

# Calculate water quality indices
bgr_result = bgr(s2_img.image, bands=['B02', 'B03'])
ndci_result = ndci(s2_img.image, bands=['B04', 'B03'])

print(f"Water quality analysis complete: BGR {bgr_result.shape}, NDCI {ndci_result.shape}")
```