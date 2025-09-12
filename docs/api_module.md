# API Module Documentation

The API module provides unified access to satellite data from multiple providers including USGS (Landsat) and Copernicus Data Space (Sentinel-2).

## Core Classes

### SatelliteProduct

Standardized representation of satellite products:

```python
from ShallowLearn.api import SatelliteProduct

product = SatelliteProduct(
    product_id="S2A_MSIL1C_20230615T103021_20230615T103020_T32UQD_20230615T124531",
    satellite="sentinel2",
    sensor="MSI",
    acquisition_date="2023-06-15",
    cloud_cover=15.2,
    processing_level="L1C",
    thumbnail_url="https://example.com/thumbnail.jpg"
)

print(f"Product: {product.product_id}")
print(f"Cloud cover: {product.cloud_cover}%")
```

### SatelliteQuery

Query parameters for satellite searches:

```python
from ShallowLearn.api import SatelliteQuery

query = SatelliteQuery(
    bbox=[-74.0, 40.7, -73.9, 40.8],  # [min_lon, min_lat, max_lon, max_lat]
    start_date="2023-06-01",
    end_date="2023-06-30",
    satellite="sentinel2",
    max_cloud_cover=20,
    processing_level="L1C"
)
```

### UnifiedSatelliteAPI

Main interface for satellite data access:

```python
from ShallowLearn.api import UnifiedSatelliteAPI
import os

# Set up environment variables first
os.environ['LSAT_TOKEN'] = 'your_usgs_token'
os.environ['LSAT_USER'] = 'your_usgs_username'
os.environ['SEN_USER'] = 'your_copernicus_username'
os.environ['SEN_PASS'] = 'your_copernicus_password'

# Initialize API
api = UnifiedSatelliteAPI()

# Search for products
products = api.search(
    bbox=[-74.0, 40.7, -73.9, 40.8],
    start_date="2023-06-01",
    end_date="2023-06-30",
    satellite="sentinel2",
    max_cloud_cover=20
)

print(f"Found {len(products)} products")
```

## Downloader Classes

### Sentinel2CDSEDownloader

Downloads Sentinel-2 data from Copernicus Data Space:

```python
from ShallowLearn.api import Sentinel2CDSEDownloader

downloader = Sentinel2CDSEDownloader()

# Download product
success = downloader.download_product(
    product_id="S2A_MSIL1C_20230615T103021_20230615T103020_T32UQD_20230615T124531",
    output_dir="./downloads"
)

if success:
    print("Download completed successfully")
```

### LandsatUSGSDownloader

Downloads Landsat data from USGS:

```python
from ShallowLearn.api import LandsatUSGSDownloader

downloader = LandsatUSGSDownloader()

# Download Landsat product
success = downloader.download_product(
    product_id="LC08_L1TP_226078_20230615_20230627_02_T1",
    output_dir="./downloads"
)
```

## Environment Setup

Create a `.env` file with required credentials:

```bash
# USGS Landsat API Credentials
LSAT_TOKEN=your_usgs_m2m_api_token_here
LSAT_USER=your_usgs_username_here

# Copernicus Sentinel API Credentials  
SEN_USER=your_copernicus_username_here
SEN_PASS=your_copernicus_password_here
```

## Complete Workflow Example

```python
from ShallowLearn.api import UnifiedSatelliteAPI
from ShallowLearn.io import Sentinel2Image
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize API
api = UnifiedSatelliteAPI()

# Search for data
products = api.search(
    bbox=[-74.0, 40.7, -73.9, 40.8],  # NYC area
    start_date="2023-06-01", 
    end_date="2023-06-30",
    satellite="sentinel2",
    max_cloud_cover=15
)

# Download first product
if products:
    product = products[0]
    print(f"Downloading: {product.product_id}")
    
    success = api.download_product(product, output_dir="./data")
    
    if success:
        # Process downloaded data
        downloaded_path = f"./data/{product.product_id}.zip"
        s2_img = Sentinel2Image(downloaded_path)
        print(f"Loaded image with shape: {s2_img.image.shape}")
```