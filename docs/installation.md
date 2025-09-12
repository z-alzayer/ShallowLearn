# Installation

Installing **ShallowLearn** is straightforward. Follow these steps to get started:

## Prerequisites

Make sure you have Python 3.10+ installed on your system. **ShallowLearn now exclusively supports [uv](https://docs.astral.sh/uv/) as the package manager**.

## Installing the Package

### Using uv (Recommended)

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/z-alzayer/ShallowLearn.git
cd ShallowLearn

# Install dependencies and the package
uv pip install -e .
```

### Environment Configuration

Create a `.env` file in the project root for API credentials:

```bash
# Copy example environment file
cp .env.example .env
# Edit with your credentials
```

Required environment variables:
- `LSAT_TOKEN` - USGS EarthExplorer M2M API token for Landsat data
- `LSAT_USER` - USGS EarthExplorer username
- `SEN_USER` - Copernicus Data Space username for Sentinel-2
- `SEN_PASS` - Copernicus Data Space password

## Development Installation

For development with additional dependencies:

```bash
uv pip install -e ".[dev]"
# This includes testing, documentation, and code quality tools
```

## System Requirements

!!! warning "Platform Support" 
This package has been tested primarily on Linux environments. While it should work on macOS and Windows, there are no guarantees.

!!! note "GDAL Dependencies"
    The package requires GDAL which can be complex to install. Using the provided environment ensures all geospatial dependencies are correctly configured.
