## Importing Modules

To utilize the helper functions contained in `ShallowLearn.ImageHelper`, import the following modules and functions:

```python
import numpy as np
import matplotlib.pyplot as plt
from ShallowLearn.ImageHelper import (
    clip_array, load_img, select_channels, plot_rgb, plot_hsv, plot_lab, plot_ycbcr,
    predict_mask, apply_mask, generate_multichannel_mask, plot_with_legend, plot_histograms
)
```

---

## Loading and Preparing Data

### Load an Image

Load a GeoTIFF image and preprocess it by clipping values:

```python
from ShallowLearn import LoadData

# Path to your GeoTIFF image
image_path = "path/to/your/image.tif"

# Load and preprocess the image
image = load_img(image_path, clip=True)
print("Image shape:", image.shape)
```

---

## Visualizing Data

### Plot an RGB Image

Visualize an RGB representation of the image:

```python
# Display RGB bands
rgb_image = plot_rgb(image, plot=True)
```

### Convert and Plot Color Spaces

Convert and plot the image in different color spaces:

```python
# Convert and plot image in HSV, LAB, and YCbCr color spaces
plot_hsv(image, plot=True)
plot_lab(image, plot=True)
plot_ycbcr(image, plot=True)
```

---

## Working with Channels

### Select Specific Channels

Extract specific channels from the image:

```python
# Select channels by their indices (e.g., 0, 1, 2)
selected_channels = select_channels(image, [0, 1, 2])
print("Selected channels shape:", selected_channels.shape)
```

### Remove a Channel

Remove a specific channel from the image:

```python
# Remove the second channel
image_without_channel = remove_channel(image, channel=1)
print("Image shape after removing channel:", image_without_channel.shape)
```

---

## Generating and Applying Masks

### Predict a Mask

Predict a mask using a pre-trained model:

```python
# Predict a mask for the image
mask = predict_mask(image, model="path/to/your/model.pkl")
```

### Apply the Mask to the Image

Mask the image and set masked areas to zero:

```python
# Apply the predicted mask
masked_image = apply_mask(image, mask, fill_value=0)
```

---

## Advanced Features

### Generate a Multichannel Mask

Create a multichannel mask for the image:

```python
# Generate multichannel mask
multichannel_mask = generate_multichannel_mask(image)
print("Multichannel mask shape:", multichannel_mask.shape)
```

### Plot Histograms

Visualize the pixel intensity distribution across channels:

```python
# Plot histograms for each channel
plot_histograms(image, bins=50, title="Pixel Intensity Histogram")
```

### Visualize Labeled Data

Plot labeled data with a legend:

```python
# Define label mapping
value_dict = {0: "Background", 1: "Vegetation", 2: "Water"}

# Visualize labels
plot_with_legend(mask, value_dict)
```

---

## Additional Utilities

### Clip Array Values

Clip array values to a specified range:

```python
# Clip values between 0 and 10,000
clipped_array = clip_array(image)
```

### Plot GeoTIFF with Coordinates

Visualize a GeoTIFF image with proper coordinate axes and a scale bar:

```python
# Example bounds: (left, right, bottom, top)
bounds = (0, 1000, 0, 1000)

# Plot the GeoTIFF image
plot_geotiff(image, bounds, title="Map with Coordinates")
```

---

## Conclusion

This guide introduces the basic and advanced features of **ShallowLearn.ImageHelper** to help you analyze geospatial and image data effectively. For more details, check the [reference guide](api_reference.md#ShallowLearn.ImageHelper) 
