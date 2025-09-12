# Segmentation Module Documentation

The Segmentation module provides superpixel generation and processing methods for image segmentation and analysis.

## Core Functions

### felzenszwalb_segmentation

Graph-based segmentation algorithm:

```python
from ShallowLearn.segmentation.superpixels import felzenszwalb_segmentation
import numpy as np

# Load or create test image
image = np.random.rand(100, 100, 3)

# Apply Felzenszwalb segmentation
segments = felzenszwalb_segmentation(
    image, 
    scale=100,      # Segmentation scale
    sigma=0.5,      # Gaussian smoothing
    min_size=50     # Minimum segment size
)

print(f"Generated {len(np.unique(segments))} segments")
```

### slic_segmentation

SLIC (Simple Linear Iterative Clustering) superpixels:

```python
from ShallowLearn.segmentation.superpixels import slic_segmentation
import numpy as np

image = np.random.rand(256, 256, 3)

# Apply SLIC segmentation
segments = slic_segmentation(
    image,
    n_segments=1000,    # Approximate number of superpixels
    compactness=10,     # Balance color vs spatial proximity
    sigma=1             # Gaussian smoothing
)

print(f"SLIC segments: {len(np.unique(segments))}")
```

### pca_segments

Apply PCA to segmented regions:

```python
from ShallowLearn.segmentation.superpixels import pca_segments
import numpy as np

# Multispectral image and segmentation
ms_image = np.random.rand(100, 100, 8)
segments = slic_segmentation(ms_image[:,:,:3], n_segments=500)

# Apply PCA to each segment
pca_result = pca_segments(
    ms_image, 
    segments, 
    n_components=3
)

print(f"PCA result shape: {pca_result.shape}")
```

### cluster_segments

Cluster segments based on spectral properties:

```python
from ShallowLearn.segmentation.superpixels import cluster_segments
import numpy as np

ms_image = np.random.rand(200, 200, 8)
segments = felzenszwalb_segmentation(ms_image[:,:,:3])

# Cluster segments using K-means
clustered = cluster_segments(
    ms_image,
    segments,
    method='kmeans',
    n_clusters=5
)

print(f"Clustered into {len(np.unique(clustered))} classes")

# DBSCAN clustering
clustered_dbscan = cluster_segments(
    ms_image,
    segments,
    method='dbscan',
    eps=0.5,
    min_samples=10
)
```

### extract_patches

Extract patches around segment centroids:

```python
from ShallowLearn.segmentation.superpixels import extract_patches
import numpy as np

image = np.random.rand(300, 300, 8)
segments = slic_segmentation(image[:,:,:3], n_segments=200)

# Extract 32x32 patches around segment centers
patches = extract_patches(
    image,
    segments,
    patch_size=(32, 32)
)

print(f"Extracted {len(patches)} patches")
print(f"Patch shape: {patches[0].shape}")
```

## Complete Superpixel Pipeline

```python
from ShallowLearn.segmentation.superpixels import (
    process_superpixel_dii_pipeline,
    slic_segmentation,
    pca_segments,
    cluster_segments
)
import numpy as np

# Simulate Sentinel-2 image
s2_image = np.random.rand(512, 512, 13)

# Complete superpixel + DII processing pipeline
result = process_superpixel_dii_pipeline(
    s2_image,
    segmentation_method='slic',
    n_segments=2000,
    n_pca_components=5,
    clustering_method='dbscan'
)

print(f"Pipeline result keys: {result.keys()}")
print(f"Segments shape: {result['segments'].shape}")
print(f"PCA features shape: {result['pca_features'].shape}")
print(f"DII values shape: {result['dii_values'].shape}")
```

## Integration with IO Module

```python
from ShallowLearn.io import Sentinel2Image
from ShallowLearn.segmentation.superpixels import slic_segmentation, pca_segments

# Load satellite image
s2_img = Sentinel2Image("sentinel2_archive.zip")

# Create RGB for segmentation
rgb_image = s2_img.get_rgb_bands()

# Generate superpixels
segments = slic_segmentation(rgb_image, n_segments=1000)

# Apply PCA to full multispectral data
pca_result = pca_segments(s2_img.image, segments, n_components=5)

# Visualize results
import matplotlib.pyplot as plt
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(rgb_image)
plt.title("Original RGB")

plt.subplot(1, 3, 2)
plt.imshow(segments, cmap='nipy_spectral')
plt.title("Superpixels")

plt.subplot(1, 3, 3)
plt.imshow(pca_result[:,:,0], cmap='viridis')
plt.title("First PCA Component")
plt.show()
```