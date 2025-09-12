# Visualization Module

The Visualization module provides plotting and display functions for remote sensing data analysis.

## Time Series Plots

### extract_point_spectra

Extract spectral values from image stack at specific coordinates:

```python
from ShallowLearn.visualization.time_series_plots import extract_point_spectra
import numpy as np

# Image stack: (time, height, width, bands)
image_stack = np.random.rand(10, 50, 50, 4)
spectra = extract_point_spectra(image_stack, x=25, y=25)
print(f"Spectra shape: {spectra.shape}")  # (10, 4)
```

### plot_spectral_timeseries  

Plot spectral values over time:

```python
from ShallowLearn.visualization.time_series_plots import plot_spectral_timeseries
import pandas as pd

# Create date index
dates = pd.date_range('2023-01-01', periods=10, freq='30D')
band_labels = {0: 'Blue', 1: 'Green', 2: 'Red', 3: 'NIR'}

fig = plot_spectral_timeseries(
    spectra=spectra,
    dates=dates, 
    band_labels=band_labels,
    title="Spectral Time Series"
)
```

### animate_images_and_timeseries

Create animated visualization with images and time series:

```python
from ShallowLearn.visualization.time_series_plots import animate_images_and_timeseries
import pandas as pd

# Sample data
images = [np.random.rand(50, 50, 3) for _ in range(5)]
dates = pd.date_range('2023-01-01', periods=5, freq='30D')
timeseries_data = pd.DataFrame({
    'B4': np.random.rand(5),
    'B3': np.random.rand(5), 
    'B2': np.random.rand(5)
}, index=dates)

# Create animation
animation_path = animate_images_and_timeseries(
    images=images,
    timeseries_data=timeseries_data,
    point_coords=(25, 25),
    output_path="animation.gif"
)
```

## QuickLook Visualization

### QuickLookVisualizer

Visualize QuickLook analysis results:

```python
from ShallowLearn.ml.quicklook_processor import QuickLookProcessor
from ShallowLearn.visualization.quicklook_viz import QuickLookVisualizer

# Run QuickLook analysis first
processor = QuickLookProcessor()
# processor.run_complete_analysis(images)  # Process your images

# Create visualizer
viz = QuickLookVisualizer(processor)
```