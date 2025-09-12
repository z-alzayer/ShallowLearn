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
import numpy as np

# Create sample image for demonstration
image = np.random.rand(50, 50, 3)

fig, ax = plt.subplots()
ax.imshow(image)
ax.set_title("Satellite Image")
standardize_axes(ax)  # Apply consistent formatting
```

## File Discovery

Find Landsat satellite files:

```python
from ShallowLearn.utilities.file_discovery import find_landsat_files

# Find Landsat files in directory
landsat_files = find_landsat_files("/path/to/data/")
print(f"Found {len(landsat_files)} Landsat files")
```

## Date Helper

Extract dates from satellite data:

```python
from ShallowLearn.utilities.date_helper import extract_dates

# Extract dates from file paths or names
dates = extract_dates(["file_20230101.tif", "file_20230201.tif"])
print(f"Extracted dates: {dates}")
```