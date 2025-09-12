# Core Module Documentation

The Core module provides fundamental array utilities and mathematical operations for image processing. These functions are lightweight, efficient building blocks for higher-level operations.

## Array Utilities

### clip_array

Clips array values to a specified range:

```python
from ShallowLearn.core.array_utils import clip_array
import numpy as np

# Landsat data with outliers
landsat_data = np.array([[-100, 500, 15000, 25000]], dtype=float)
clipped_data = clip_array(landsat_data, min_val=0, max_val=10000)
print(clipped_data)  # [[0, 500, 10000, 10000]]

# Reflectance values to 0-1 range
reflectance_data = np.random.normal(0.3, 0.5, (100, 100, 8))
reflectance_clipped = clip_array(reflectance_data, min_val=0, max_val=1)
```

### select_channels

Selects specific channels from multispectral imagery:

```python
from ShallowLearn.core.array_utils import select_channels
import numpy as np

# 13-band Sentinel-2 image
s2_image = np.random.rand(256, 256, 13)

# RGB channels (Red=3, Green=2, Blue=1)
rgb_indices = [3, 2, 1]  # B04, B03, B02
rgb_image = select_channels(s2_image, rgb_indices)
print(f"RGB shape: {rgb_image.shape}")  # (256, 256, 3)

# False color composite (NIR, Red, Green)
false_color_indices = [7, 3, 2]  # B08, B04, B03
false_color = select_channels(s2_image, false_color_indices)
```

### remove_channel

Removes specific channels from multispectral data:

```python
from ShallowLearn.core.array_utils import remove_channel
import numpy as np

s2_image = np.random.rand(100, 100, 13)

# Remove cirrus band (index 10)
no_cirrus = remove_channel(s2_image, 10)
print(f"After removing cirrus: {no_cirrus.shape}")  # (100, 100, 12)
```

### normalize_array

Normalizes arrays using different scaling methods:

```python
from ShallowLearn.core.array_utils import normalize_array
import numpy as np

ms_data = np.random.rand(50, 50, 8) * 10000

# Min-max normalization (0-1 range)
normalized_minmax = normalize_array(ms_data, method='minmax')

# Z-score normalization (mean=0, std=1)
normalized_zscore = normalize_array(ms_data, method='standard')

# Robust scaling (less sensitive to outliers)
normalized_robust = normalize_array(ms_data, method='robust')
```

### apply_mask

Applies boolean masks to imagery data:

```python
from ShallowLearn.core.array_utils import apply_mask
import numpy as np

image_data = np.random.rand(100, 100, 8)
cloud_mask = np.random.random((100, 100)) > 0.8  # 20% clouds

# Apply cloud mask (set cloudy pixels to NaN)
masked_image = apply_mask(image_data, cloud_mask, fill_value=np.nan)

# Count valid pixels
valid_pixels = np.sum(~np.isnan(masked_image[:, :, 0]))
print(f"Valid pixels: {valid_pixels}")
```

### get_band_numbers

Converts band names to numerical indices:

```python
from ShallowLearn.core.array_utils import get_band_numbers

s2_bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 
            'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']

# Get indices for RGB bands
rgb_band_names = ['B04', 'B03', 'B02']
rgb_indices = get_band_numbers(rgb_band_names, s2_bands)
print(f"RGB indices: {rgb_indices}")  # [3, 2, 1]
```

## Complete Preprocessing Example

```python
from ShallowLearn.core.array_utils import *
import numpy as np

# Raw Sentinel-2 data
raw_s2 = np.random.rand(512, 512, 13) * 10000

# 1. Clip to valid range
clipped_s2 = clip_array(raw_s2, min_val=0, max_val=10000)

# 2. Remove atmospheric bands
clean_s2 = remove_channel(clipped_s2, [9, 10])

# 3. Normalize to reflectance
reflectance_s2 = normalize_array(clean_s2, method='minmax')

# 4. Create and apply cloud mask
cirrus_band = raw_s2[:, :, 10]
cloud_mask = cirrus_band > 500
masked_s2 = apply_mask(reflectance_s2, cloud_mask, fill_value=np.nan)

print(f"Final shape: {masked_s2.shape}")
```