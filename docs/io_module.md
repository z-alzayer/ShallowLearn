# IO Module

The IO module provides interfaces for loading and processing satellite imagery data. It supports both Landsat and Sentinel-2 imagery with automatic band detection and metadata preservation.

## Core Classes

### Sentinel2Image

Handles Sentinel-2 archives with comprehensive band support:

```python
from ShallowLearn.io.satellite_data import Sentinel2Image

# Load Sentinel-2 image
s2_img = Sentinel2Image("path/to/sentinel2.zip")

# Check available bands
print(f"Available bands: {s2_img.present_bands}")
print(f"Image shape: {s2_img.image.shape}")

# Get RGB band names
rgb_bands = s2_img.get_rgb_bands()
print(f"RGB bands: {rgb_bands}")  # ('B04', 'B03', 'B02')

# Get specific band data
red_data = s2_img.get_band_data('B04')  # Red band
green_data = s2_img.get_band_data('B03')  # Green band
blue_data = s2_img.get_band_data('B02')  # Blue band
```

**Sentinel-2 Band Information:**
- B02: Blue (10m resolution)
- B03: Green (10m resolution)  
- B04: Red (10m resolution)
- B08: NIR (10m resolution)
- B05: Red Edge (20m resolution)
- B11, B12: SWIR (20m resolution)

### LandsatImage

Handles Landsat archives:

```python
from ShallowLearn.load.landsatdata_class import LandSatImage

# Load Landsat image
landsat_img = LandSatImage("path/to/landsat.tar")
print(f"Available bands: {landsat_img.present_bands}")
```

## Metadata Access

```python
# Get image metadata
metadata = s2_img.get_metadata()
print(f"Processing level: {metadata.get('processing_level', 'Unknown')}")

# Check image bounds
bounds = s2_img.get_bounds()
print(f"Image bounds: {bounds}")
```

## Integration with Features Module

Use IO module with spectral indices:

```python
from ShallowLearn.io.satellite_data import Sentinel2Image
from ShallowLearn.features.indices import bgr, ndci

# Load image
s2_img = Sentinel2Image("sentinel2.zip")

# Calculate water quality indices
bgr_result = bgr(s2_img.image, bands=['B02', 'B03'])  # Blue-Green Ratio
ndci_result = ndci(s2_img.image, bands=['B04', 'B03'])  # Chlorophyll index
```