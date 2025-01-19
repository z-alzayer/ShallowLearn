
This guide covers the basic usage of the `ShallowLearn.ComputerVisionFeatures` module. These functions are designed for tasks such as edge detection, texture analysis, and feature extraction. Follow the steps below to understand and utilize these tools effectively.

---

## Installation

Ensure you have the required dependencies installed. Use the following commands to install them:

```bash
pip install numpy opencv-python scikit-image
```

---

## Importing Modules

To use the functions, import the `ShallowLearn.ComputerVisionFeatures` module:

```python
import numpy as np
import cv2
from ShallowLearn.ComputerVisionFeatures import (
    edge_density,
    texture_features,
    color_histogram,
    sobel_edge_detection,
    gabor_features,
    histogram_of_oriented_gradients
)
```

---

## Edge Detection

### Edge Density

Computes the edge density of an image using the Canny Edge Detector.

```python
# Example usage
edge_density_map = edge_density(image)
```

---

## Texture Analysis

### Local Binary Pattern (LBP)

Computes texture features using the Local Binary Pattern method.

```python
# Example usage
lbp_texture = texture_features(image, P=8, R=1)
```

---

## Color Analysis

### Color Histogram

Computes the color histogram for each channel in the image.

```python
# Example usage
hist = color_histogram(image, bins=32)
```

---

## Feature Extraction

### Histogram of Oriented Gradients (HOG)

Computes the HOG feature descriptor for an image.

```python
# Example usage
hog_features = histogram_of_oriented_gradients(image, pixels_per_cell=(16, 16), cells_per_block=(4, 4), orientations=9)
```

---

## Advanced Edge Detection

### Sobel Edge Detection

Applies Sobel edge detection to an image.

```python
# Example usage
sobel_edges = sobel_edge_detection(image)
```

---

## Gabor Filtering

### Gabor Features

Applies a Gabor filter to an image.

```python
# Example usage
gabor_response = gabor_features(image, frequency=0.6)
```

---

## Conclusion

This guide introduces the core functions in `ShallowLearn.ComputerVisionFeatures` to help analyze and process image data effectively. Explore each function and adapt them to your specific needs. For more detailed documentation, check the [reference guide](api_reference.md#ShallowLearn.ComputerVisionFeatures) 