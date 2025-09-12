# Development Roadmap

Based on the current codebase analysis and documentation review, here are the key changes needed for future development.

## Critical IO Module Improvements

### 1. Built-in Band Math Methods

The IO classes need convenience methods for common calculations instead of requiring manual numpy operations:

```python
# Current (too much numpy required)
red = s2_img.get_band('B04')
nir = s2_img.get_band('B08')
ndvi = np.where((red + nir) != 0, (nir - red) / (nir + red), np.nan)

# Needed
ndvi = s2_img.calculate_ndvi()
evi = s2_img.calculate_evi()
ndwi = s2_img.calculate_ndwi()
```

**Required methods for SatelliteImage classes:**
- `calculate_ndvi()` - Normalized Difference Vegetation Index
- `calculate_evi()` - Enhanced Vegetation Index  
- `calculate_ndwi()` - Normalized Difference Water Index
- `calculate_mndwi()` - Modified NDWI
- `calculate_ndbi()` - Normalized Difference Built-up Index
- `get_rgb_bands()` - Return RGB array ready for visualization
- `get_false_color()` - Return false color composite
- `apply_cloud_mask()` - Apply cloud masking based on available bands

### 2. Improved Band Access

```python
# Current
band = s2_img.get_band('B04')

# Needed convenience methods
bands = s2_img.get_bands(['B04', 'B03', 'B02'])  # Multiple bands
reflectance = s2_img.to_reflectance()  # Auto-convert to 0-1 range
rgb = s2_img.to_rgb(bands=[4, 3, 2])  # Direct RGB conversion
```

### 3. Metadata Enhancement

Better metadata access and preservation:
```python
# Needed
print(s2_img.acquisition_date)
print(s2_img.cloud_cover_percentage)
print(s2_img.solar_azimuth)
print(s2_img.coordinate_system)
```

## Superpixel Method Integration

Currently superpixel methods are scattered. Need to consolidate into segmentation module:

### Files to Refactor:
- `SuperPixelExtraction.py` → `segmentation/superpixels.py` (extend existing)
- `SuperPixelProcessing.py` → `segmentation/processing.py` (new)
- `SuperPixel.py` → `segmentation/legacy.py` (compatibility)

### Integration Plan:
```python
# Target API
from ShallowLearn.segmentation import SuperPixelProcessor

processor = SuperPixelProcessor(method='slic')
segments = processor.segment(s2_img.image)
features = processor.extract_features(s2_img.image, segments)
clusters = processor.cluster_segments(features)
```

## API Module Enhancements

### 1. Environment Variable Consistency

Current `.env.example` uses:
- `LSAT_TOKEN`, `LSAT_USER` 
- `SEN_USER`, `SEN_PASS`

But documentation and some code expects:
- `USGS_USERNAME`, `USGS_PASSWORD`
- `CDSE_USERNAME`, `CDSE_PASSWORD`

**Action:** Standardize on the `.env.example` format throughout.

### 2. QuickLook Integration

Better integration between API and ML modules:
```python
# Target workflow
api = UnifiedSatelliteAPI()
products = api.search(bbox=bbox, max_cloud_cover=20)

# Apply QuickLook filtering directly
filtered = api.apply_quicklook_filter(products, config=quicklook_config)
```

## Core Module Completion

Several functions referenced in tests are missing implementations:

### Missing Functions in `core/array_utils.py`:
- `safe_divide()` - Division with fill values for zero denominators
- `calculate_stats()` - Comprehensive statistics with percentiles
- `apply_function_blockwise()` - Memory-efficient processing
- `rgb_to_hsv()`, `hsv_to_rgb()` - Color space conversions
- `rgb_to_lab()`, `lab_to_rgb()` - LAB color space support

## Testing Infrastructure

### Required Test Additions:
1. **Integration tests** between modules
2. **Real data tests** with actual satellite files
3. **Performance benchmarks** for large datasets
4. **API credential validation** tests

### Test Data Organization:
- Create `tests/data/` directory with sample files
- Add synthetic data generators for consistent testing
- Mock API responses for offline testing

## Documentation Improvements

### 1. Working Examples

All code examples in documentation need validation. Currently many examples use placeholder data that may not work with actual ShallowLearn functions.

### 2. Jupyter Notebook Tutorials

Create example notebooks for:
- Basic satellite data loading and visualization
- Water quality analysis workflow
- Superpixel-based analysis
- Multi-temporal change detection
- Integration with external tools (QGIS, etc.)

### 3. Performance Guidelines

Document memory usage and processing time expectations for different dataset sizes.

## Package Structure Modernization

### 1. Move to `pyproject.toml` Only

Currently using both `setup.py` and `pyproject.toml`. Consolidate to `pyproject.toml` only with uv.

### 2. Optional Dependencies

Structure dependencies better:
```toml
[project.optional-dependencies]
ml = ["umap-learn", "scikit-learn>=1.0"]
api = ["cdsetool", "requests-oauthlib"]
viz = ["matplotlib", "seaborn"]
full = ["umap-learn", "cdsetool", "requests-oauthlib", "matplotlib", "seaborn"]
```

### 3. Entry Points

Add CLI entry points for common operations:
```bash
shallow-learn download --bbox "..." --date-range "..."
shallow-learn quicklook --input-dir "..." --output-dir "..."
```

## Visualization Enhancements

### Missing Visualization Features:
- Interactive plots with plotly/bokeh
- Multi-band image composites with automatic scaling
- Time series plotting for temporal analysis
- Histogram matching visualization
- Before/after comparison plots

## Code Quality Improvements

### 1. Type Hints

Add comprehensive type hints throughout codebase:
```python
from typing import List, Dict, Optional, Union, Tuple
import numpy.typing as npt

def calculate_ndvi(
    red: npt.NDArray[np.float64], 
    nir: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
```

### 2. Error Handling

Standardize error handling patterns:
```python
class ShallowLearnError(Exception):
    """Base exception for ShallowLearn"""
    pass

class BandNotFoundError(ShallowLearnError):
    """Raised when required band is not available"""
    pass
```

### 3. Logging

Add proper logging throughout:
```python
import logging
logger = logging.getLogger(__name__)

logger.info(f"Loading satellite image: {file_path}")
logger.warning(f"Band B10 missing, using interpolation")
```

## Priority Implementation Order

1. **High Priority:**
   - IO module band math methods
   - Core module missing functions
   - Test data and validation
   - Working documentation examples

2. **Medium Priority:**
   - Superpixel consolidation
   - API module improvements
   - Type hints and error handling

3. **Low Priority:**
   - CLI entry points
   - Interactive visualizations
   - Advanced ML features

## Breaking Changes to Consider

### Version 2.0 Planning:
- Remove legacy functions and maintain only modular structure
- Standardize all band indexing (0-based vs 1-based)
- Unify coordinate system handling
- Simplify configuration management

This roadmap should guide development priorities and ensure the package becomes more user-friendly and maintainable.