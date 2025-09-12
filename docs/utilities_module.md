# Utilities Module

The Utilities module provides helper functions and support tools used throughout ShallowLearn.

## Core Utilities

### clip_image

Apply percentile clipping to enhance image contrast:

```python
from ShallowLearn.utilities.util import clip_image
import numpy as np

# Create sample image
image = np.random.rand(100, 100, 3) * 255

# Apply 2% clipping
clipped = clip_image(image, clip_percent=2.0)
print(f"Original range: {image.min():.1f} - {image.max():.1f}")
print(f"Clipped range: {clipped.min():.3f} - {clipped.max():.3f}")
```

### extract_point_spectra

Extract spectral time series from image stack:

```python  
from ShallowLearn.utilities.util import extract_point_spectra
import numpy as np

# Image stack: (time, height, width, bands) 
image_stack = np.random.rand(10, 50, 50, 4)
spectra = extract_point_spectra(image_stack, x=25, y=25)
print(f"Extracted spectra shape: {spectra.shape}")  # (10, 4)
```

### standardize_axes

Apply consistent matplotlib formatting:

```python
from ShallowLearn.utilities.util import standardize_axes  
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.imshow(image)
ax.set_title("Satellite Image")
standardize_axes(ax)  # Apply consistent formatting
```

## File Discovery

Find satellite imagery files:

```python
from ShallowLearn.utilities.file_discovery import find_satellite_files

# Find Sentinel-2 files
s2_files = find_satellite_files(
    directory="/data/satellite/",
    sensor_type="sentinel2", 
    recursive=True
)
print(f"Found {len(s2_files)} Sentinel-2 scenes")
```

## Date Helper

Handle satellite imagery dates:

```python
from ShallowLearn.utilities.date_helper import parse_satellite_date

# Parse Sentinel-2 filename
filename = "S2A_MSIL2A_20230315T103021_N0509_R108_T32TNR_20230315T143534.SAFE"
date = parse_satellite_date(filename)
print(f"Acquisition date: {date}")
```