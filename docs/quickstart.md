
# Quick Start Guide

Get started with ShallowLearn's core functionality for satellite data analysis.

## Installation

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install ShallowLearn
git clone https://github.com/z-alzayer/ShallowLearn.git
cd ShallowLearn
uv pip install -e .
```

## Basic Usage

### Spectral Indices

```python
from ShallowLearn.features.indices import bgr, ndci
import numpy as np

# Create sample 4-band image
image = np.random.rand(100, 100, 4) * 0.3

# Calculate Blue-Green Ratio for water quality
bgr_result = bgr(image, bands=['B02', 'B03'])
print(f"BGR shape: {bgr_result.shape}")

# Calculate chlorophyll index
ndci_result = ndci(image, bands=['B04', 'B03'])
print(f"NDCI shape: {ndci_result.shape}")
```

### Time Series Visualization

```python
from ShallowLearn.visualization.time_series_plots import plot_spectral_timeseries
import pandas as pd
import numpy as np

# Sample spectral time series
spectra = np.random.rand(10, 4) * 0.5
dates = pd.date_range('2023-01-01', periods=10, freq='30D')
band_labels = {0: 'Blue', 1: 'Green', 2: 'Red', 3: 'NIR'}

# Plot time series
fig = plot_spectral_timeseries(
    spectra=spectra,
    dates=dates, 
    band_labels=band_labels,
    title="Spectral Evolution"
)
```

### QuickLook Analysis

```python
from ShallowLearn.ml.quicklook_processor import QuickLookProcessor

# Initialize processor
processor = QuickLookProcessor()

# Process satellite images (with your actual image paths)
# results = processor.run_complete_analysis(image_paths)
```

## Next Steps

- **[IO Module](io_module.md)** - Detailed satellite data loading
- **[ML Module](ml_module.md)** - Machine learning workflows  
- **[Features Module](features_module.md)** - Spectral index calculations
- **[Visualization Module](visualization_module.md)** - Plotting functions
- **[API Reference](api_reference.md)** - Complete function documentation
