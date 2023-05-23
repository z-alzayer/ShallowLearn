import numpy as np
from ShallowLearn.band_mapping import band_mapping

def ci(image, bands=None):
    """
    Computes the Chlorophyll Index (CI) using the specified band codes.
    """
    if bands is None:
        bands = ['B04', 'B05', 'B08']  # Default band codes
    band_numbers = get_band_numbers(bands)
    validate_band_shape(image, band_numbers)
    red_band = image[:, :, band_numbers[0]]
    red_edge_band = image[:, :, band_numbers[1]]
    nir_band = image[:, :, band_numbers[2]]
    ci = (nir_band - red_band) / (nir_band + red_band + red_edge_band)
    return ci


def oci(image, bands=None):
    """
    Computes the Ocean Color Index (OCI) using the specified band codes.
    """
    if bands is None:
        bands = ['B03', 'B12']  # Default band codes
    band_numbers = get_band_numbers(bands)
    validate_band_shape(image, band_numbers)
    green_band = image[:, :, band_numbers[0]].astype(float)
    swir_band = image[:, :, band_numbers[1]].astype(float)
    
    # Avoid division by zero
    swir_band[swir_band == 0] = np.nan
    
    oci = (green_band / swir_band) - 1
    return oci

def ssi(image, bands=None):
    """
    Computes the Suspended Sediment Index (SSI) using the specified band codes.
    """
    if bands is None:
        bands = ['B04', 'B05', 'B08']  # Default band codes
    band_numbers = get_band_numbers(bands)
    validate_band_shape(image, band_numbers)
    red_band = image[:, :, band_numbers[0]]
    red_edge_band = image[:, :, band_numbers[1]]
    nir_band = image[:, :, band_numbers[2]]
    ssi = (nir_band - red_band) / (nir_band + red_band + red_edge_band)
    return ssi

def ti(image, bands=None):
    """
    Computes the Turbidity Index (TI) using the specified band codes.
    """
    if bands is None:
        bands = ['B02', 'B04', 'B05', 'B12']  # Default band codes
    band_numbers = get_band_numbers(bands)
    validate_band_shape(image, band_numbers)
    green_band = image[:, :, band_numbers[0]]
    red_band = image[:, :, band_numbers[1]]
    red_edge_band = image[:, :, band_numbers[2]]
    swir_band = image[:, :, band_numbers[3]]
    ti = (green_band + red_band + red_edge_band) / swir_band
    return ti

def wqi(image, bands=None):
    """
    Computes the Water Quality Index (WQI) using the specified band codes.
    """
    if bands is None:
        bands = ['B03', 'B04', 'B05', 'B06']  # Default band codes
    band_numbers = get_band_numbers(bands)
    validate_band_shape(image, band_numbers)
    blue_band = image[:, :, band_numbers[0]]
    green_band = image[:, :, band_numbers[1]]
    red_band = image[:, :, band_numbers[2]]
    nir_band = image[:, :, band_numbers[3]]
    wqi = (blue_band + green_band + red_band + nir_band) / 4
    return wqi

def ndci(image, bands=None):
    """
    Computes the Normalized Difference Chlorophyll Index (NDCI) using the specified band codes.
    """
    if bands is None:
        bands = ['B03', 'B05']  # Default band codes
    band_numbers = get_band_numbers(bands)
    validate_band_shape(image, band_numbers)
    green_band = image[:, :, band_numbers[0]]
    red_edge_band = image[:, :, band_numbers[1]]
    ndci = (green_band - red_edge_band) / (green_band + red_edge_band)
    return ndci

def get_band_numbers(bands):
    """
    Converts band codes to band numbers using the band mapping dictionary.
    """
    return [band_mapping[band]['index'] for band in bands]

def validate_band_shape(image, bands):
    """
    Validates that the bands have the expected shape in the image array.
    """
    if len(image.shape) < 3 or image.shape[2] < max(bands) + 1:
        raise ValueError("Invalid band shape in the image array.")