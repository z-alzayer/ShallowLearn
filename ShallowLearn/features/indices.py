"""
Spectral indices for water quality and marine remote sensing.
Clean implementation with configurable band mapping.
"""
import numpy as np
from typing import List, Dict, Optional, Union
from functools import wraps

# Default Sentinel-2 band mapping for backward compatibility
DEFAULT_SENTINEL2_MAPPING = {
    'B01': {'index': 0, 'name': 'Coastal/Aerosol', 'wavelength': 443},
    'B02': {'index': 1, 'name': 'Blue', 'wavelength': 490},
    'B03': {'index': 2, 'name': 'Green', 'wavelength': 560},
    'B04': {'index': 3, 'name': 'Red', 'wavelength': 665},
    'B05': {'index': 4, 'name': 'RedEdge1', 'wavelength': 705},
    'B06': {'index': 5, 'name': 'RedEdge2', 'wavelength': 740},
    'B07': {'index': 6, 'name': 'RedEdge3', 'wavelength': 783},
    'B08': {'index': 7, 'name': 'NIR', 'wavelength': 842},
    'B8A': {'index': 8, 'name': 'NIR narrow', 'wavelength': 865},
    'B09': {'index': 9, 'name': 'Water vapour', 'wavelength': 945},
    'B10': {'index': 10, 'name': 'Cirrus', 'wavelength': 1375},
    'B11': {'index': 11, 'name': 'SWIR1', 'wavelength': 1610},
    'B12': {'index': 12, 'name': 'SWIR2', 'wavelength': 2190}
}



def get_band_numbers(bands: List[str], band_mapping: Dict) -> List[int]:
    """
    Converts band names to band numbers using a band mapping dictionary.

    Parameters:
    -----------
    bands : List[str]
        List of band names (e.g., ['B02', 'B03', 'B04'])
    band_mapping : Dict
        Dictionary mapping band names to indices

    Returns:
    --------
    List[int]
        List of band indices

    Raises:
    -------
    KeyError
        If a band name is not found in the mapping
    """
    band_numbers = []
    for band in bands:
        if band not in band_mapping:
            raise KeyError(f"Band '{band}' not found in band mapping")
        band_numbers.append(band_mapping[band]['index'])
    return band_numbers


def validate_band_shape(image: np.ndarray, band_numbers: List[int]) -> None:
    """
    Validates that the image has enough bands for the given band numbers.

    Parameters:
    -----------
    image : np.ndarray
        Input image array with shape (height, width, bands)
    band_numbers : List[int]
        List of band indices to validate

    Raises:
    -------
    ValueError
        If any band number is out of bounds
    """
    if len(image.shape) != 3:
        raise ValueError("Image must be 3D array with shape (height, width, bands)")
    
    max_band = max(band_numbers) if band_numbers else -1
    if max_band >= image.shape[2]:
        raise ValueError(f"Band index {max_band} is out of bounds for image with {image.shape[2]} bands")


def chlorophyll_index(image: np.ndarray, 
                     bands: Optional[List[str]] = None,
                     band_mapping: Optional[Dict] = None) -> np.ndarray:
    """
    Computes the Chlorophyll Index (CI) using the specified band codes.
    
    Purpose:
    The Chlorophyll Index is used to estimate chlorophyll content in vegetation. 
    It provides information about the health and vigor of plants.

    Reference: 
    https://www.tandfonline.com/doi/pdf/10.1080/0143116042000274015
    
    Parameters:
    -----------
    image : np.ndarray
        The input image array with shape (height, width, num_bands)
    bands : List[str], optional
        The band codes to use for computing the index. Default is ['B04', 'B05', 'B06', 'B07']
    band_mapping : Dict, optional
        Band mapping dictionary. Uses Sentinel-2 mapping if None.
    
    Returns:
    --------
    np.ndarray
        The computed Chlorophyll Index with shape (height, width)
    """
    if bands is None:
        bands = ['B04', 'B05', 'B06', 'B07']  # Default band codes
    if band_mapping is None:
        band_mapping = DEFAULT_SENTINEL2_MAPPING
        
    band_numbers = get_band_numbers(bands, band_mapping)
    validate_band_shape(image, band_numbers)
    
    r665 = image[:, :, band_numbers[0]].astype(float)
    r705 = image[:, :, band_numbers[1]].astype(float)
    r740 = image[:, :, band_numbers[2]].astype(float)
    r783 = image[:, :, band_numbers[3]].astype(float)

    ri = r665 + r783
    # Avoid division by zero
    denominator = (r740 + r705) + 1
    rep = 700 + 40 * (ri - r705) / denominator
    
    return rep


def ocean_color_index(image: np.ndarray, 
                     bands: Optional[List[str]] = None,
                     band_mapping: Optional[Dict] = None) -> np.ndarray:
    """
    Computes the Ocean Color Index (OCI) using the specified band codes.
    
    Purpose:
    The Ocean Color Index is used to assess ocean color properties, particularly 
    the presence of chlorophyll. It helps in studying phytoplankton abundance and 
    water quality in marine environments.
    
    Reference: 
    https://www.sciencedirect.com/science/article/pii/S0034425709001710

    Parameters:
    -----------
    image : np.ndarray
        The input image array with shape (height, width, num_bands)
    bands : List[str], optional
        The band codes to use for computing the index. Default is ['B04','B08','B12']
    band_mapping : Dict, optional
        Band mapping dictionary. Uses Sentinel-2 mapping if None.
    
    Returns:
    --------
    np.ndarray
        The computed Ocean Color Index with shape (height, width)
    """
    if bands is None:
        bands = ['B04', 'B08', 'B12']  # Default band codes
    if band_mapping is None:
        band_mapping = DEFAULT_SENTINEL2_MAPPING
        
    band_numbers = get_band_numbers(bands, band_mapping)
    validate_band_shape(image, band_numbers)
    
    red = image[:, :, band_numbers[0]].astype(float)
    nir = image[:, :, band_numbers[1]].astype(float)
    swir = image[:, :, band_numbers[2]].astype(float)
    
    # Floating Algae Index (FAI) calculation
    fai = red + (swir - red) * (nir - red) / ((swir - red) + 1)  # Avoid division by zero
    
    return fai


def chlorophyll_ocean_color_index(image: np.ndarray, 
                                 bands: Optional[List[str]] = None,
                                 band_mapping: Optional[Dict] = None) -> np.ndarray:
    """
    Computes the Chlorophyll Ocean Color Index.
    
    Reference: 
    https://agupubs.onlinelibrary.wiley.com/doi/pdfdirect/10.1029/2019JC015498

    Parameters:
    -----------
    image : np.ndarray
        The input image array with shape (height, width, num_bands)
    bands : List[str], optional
        The band codes to use. Default is ['B02','B03','B04']
    band_mapping : Dict, optional
        Band mapping dictionary. Uses Sentinel-2 mapping if None.
    
    Returns:
    --------
    np.ndarray
        The computed Chlorophyll Ocean Color Index
    """
    if bands is None:
        bands = ['B02', 'B03', 'B04']
    if band_mapping is None:
        band_mapping = DEFAULT_SENTINEL2_MAPPING
        
    band_numbers = get_band_numbers(bands, band_mapping)
    validate_band_shape(image, band_numbers)
    
    blue = image[:, :, band_numbers[0]].astype(float)
    green = image[:, :, band_numbers[1]].astype(float)
    red = image[:, :, band_numbers[2]].astype(float)
    
    # Chlorophyll-a concentration estimation
    ratio = blue / (green + 1)  # Avoid division by zero
    
    return ratio


def suspended_sediment_index(image: np.ndarray, 
                           bands: Optional[List[str]] = None,
                           band_mapping: Optional[Dict] = None) -> np.ndarray:
    """
    Computes the Suspended Sediment Index (SSI).

    Parameters:
    -----------
    image : np.ndarray
        The input image array with shape (height, width, num_bands)
    bands : List[str], optional
        The band codes to use. Default is ['B04', 'B05', 'B08']
    band_mapping : Dict, optional
        Band mapping dictionary. Uses Sentinel-2 mapping if None.
    
    Returns:
    --------
    np.ndarray
        The computed Suspended Sediment Index
    """
    if bands is None:
        bands = ['B04', 'B05', 'B08']
    if band_mapping is None:
        band_mapping = DEFAULT_SENTINEL2_MAPPING
        
    band_numbers = get_band_numbers(bands, band_mapping)
    validate_band_shape(image, band_numbers)
    
    red = image[:, :, band_numbers[0]].astype(float)
    red_edge = image[:, :, band_numbers[1]].astype(float)
    nir = image[:, :, band_numbers[2]].astype(float)
    
    # SSI calculation
    ssi = (red * nir) / (red_edge + 1)  # Avoid division by zero
    
    return ssi


def turbidity_index(image: np.ndarray, 
                   bands: Optional[List[str]] = None,
                   band_mapping: Optional[Dict] = None) -> np.ndarray:
    """
    Computes the Turbidity Index (TI).

    Parameters:
    -----------
    image : np.ndarray
        The input image array with shape (height, width, num_bands)
    bands : List[str], optional
        The band codes to use. Default is ['B04', 'B08']
    band_mapping : Dict, optional
        Band mapping dictionary. Uses Sentinel-2 mapping if None.
    
    Returns:
    --------
    np.ndarray
        The computed Turbidity Index
    """
    if bands is None:
        bands = ['B04', 'B08']
    if band_mapping is None:
        band_mapping = DEFAULT_SENTINEL2_MAPPING
        
    band_numbers = get_band_numbers(bands, band_mapping)
    validate_band_shape(image, band_numbers)
    
    red = image[:, :, band_numbers[0]].astype(float)
    nir = image[:, :, band_numbers[1]].astype(float)
    
    # Simple turbidity calculation
    ti = red / (nir + 1)  # Avoid division by zero
    
    return ti


def water_quality_index(image: np.ndarray, 
                       bands: Optional[List[str]] = None,
                       band_mapping: Optional[Dict] = None) -> np.ndarray:
    """
    Computes the Water Quality Index (WQI).

    Parameters:
    -----------
    image : np.ndarray
        The input image array with shape (height, width, num_bands)
    bands : List[str], optional
        The band codes to use. Default is ['B03', 'B04', 'B05', 'B06']
    band_mapping : Dict, optional
        Band mapping dictionary. Uses Sentinel-2 mapping if None.
    
    Returns:
    --------
    np.ndarray
        The computed Water Quality Index
    """
    if bands is None:
        bands = ['B03', 'B04', 'B05', 'B06']
    if band_mapping is None:
        band_mapping = DEFAULT_SENTINEL2_MAPPING
        
    band_numbers = get_band_numbers(bands, band_mapping)
    validate_band_shape(image, band_numbers)
    
    green = image[:, :, band_numbers[0]].astype(float)
    red = image[:, :, band_numbers[1]].astype(float)
    red_edge1 = image[:, :, band_numbers[2]].astype(float)
    red_edge2 = image[:, :, band_numbers[3]].astype(float)
    
    # Composite water quality index
    wqi = (green + red_edge1) / (red + red_edge2 + 1)  # Avoid division by zero
    
    return wqi


def normalized_difference_chlorophyll_index(image: np.ndarray, 
                                          bands: Optional[List[str]] = None,
                                          band_mapping: Optional[Dict] = None) -> np.ndarray:
    """
    Computes the Normalized Difference Chlorophyll Index (NDCI).

    Parameters:
    -----------
    image : np.ndarray
        The input image array with shape (height, width, num_bands)
    bands : List[str], optional
        The band codes to use. Default is ['B03', 'B05']
    band_mapping : Dict, optional
        Band mapping dictionary. Uses Sentinel-2 mapping if None.
    
    Returns:
    --------
    np.ndarray
        The computed NDCI
    """
    if bands is None:
        bands = ['B03', 'B05']
    if band_mapping is None:
        band_mapping = DEFAULT_SENTINEL2_MAPPING
        
    band_numbers = get_band_numbers(bands, band_mapping)
    validate_band_shape(image, band_numbers)
    
    green = image[:, :, band_numbers[0]].astype(float)
    red_edge = image[:, :, band_numbers[1]].astype(float)
    
    # NDCI calculation
    numerator = red_edge - green
    denominator = red_edge + green + 1  # Avoid division by zero
    ndci = numerator / denominator
    
    return ndci


def cloud_index(image: np.ndarray, 
               bands: Optional[List[str]] = None,
               band_mapping: Optional[Dict] = None) -> np.ndarray:
    """
    Computes a cloud detection index.

    Parameters:
    -----------
    image : np.ndarray
        The input image array with shape (height, width, num_bands)
    bands : List[str], optional
        The band codes to use. Default is ['B08', 'B11']
    band_mapping : Dict, optional
        Band mapping dictionary. Uses Sentinel-2 mapping if None.
    
    Returns:
    --------
    np.ndarray
        The computed cloud index
    """
    if bands is None:
        bands = ['B08', 'B11']
    if band_mapping is None:
        band_mapping = DEFAULT_SENTINEL2_MAPPING
        
    band_numbers = get_band_numbers(bands, band_mapping)
    validate_band_shape(image, band_numbers)
    
    nir = image[:, :, band_numbers[0]].astype(float)
    swir1 = image[:, :, band_numbers[1]].astype(float)
    
    # Simple cloud index
    cloud_idx = (nir - swir1) / (nir + swir1 + 1)  # Avoid division by zero
    
    return cloud_idx


def blue_green_ratio(image: np.ndarray, 
                    bands: Optional[List[str]] = None,
                    band_mapping: Optional[Dict] = None) -> np.ndarray:
    """
    Computes the Blue to Green Ratio (BGR).

    Parameters:
    -----------
    image : np.ndarray
        The input image array with shape (height, width, num_bands)
    bands : List[str], optional
        The band codes to use. Default is ['B02', 'B03']
    band_mapping : Dict, optional
        Band mapping dictionary. Uses Sentinel-2 mapping if None.
    
    Returns:
    --------
    np.ndarray
        The computed BGR
    """
    if bands is None:
        bands = ['B02', 'B03']
    if band_mapping is None:
        band_mapping = DEFAULT_SENTINEL2_MAPPING
        
    band_numbers = get_band_numbers(bands, band_mapping)
    validate_band_shape(image, band_numbers)
    
    blue = image[:, :, band_numbers[0]].astype(float)
    green = image[:, :, band_numbers[1]].astype(float)
    
    # BGR calculation
    bgr = blue / (green + 1)  # Avoid division by zero
    
    return bgr


def water_surface_index(image: np.ndarray, 
                       bands: Optional[List[str]] = None,
                       band_mapping: Optional[Dict] = None) -> np.ndarray:
    """
    Computes a water surface detection index.

    Parameters:
    -----------
    image : np.ndarray
        The input image array with shape (height, width, num_bands)
    bands : List[str], optional
        The band codes to use. Default is ['B03', 'B04', 'B08', 'B11', 'B12']
    band_mapping : Dict, optional
        Band mapping dictionary. Uses Sentinel-2 mapping if None.
    
    Returns:
    --------
    np.ndarray
        The computed water surface index
    """
    if bands is None:
        bands = ['B03', 'B04', 'B08', 'B11', 'B12']
    if band_mapping is None:
        band_mapping = DEFAULT_SENTINEL2_MAPPING
        
    band_numbers = get_band_numbers(bands, band_mapping)
    validate_band_shape(image, band_numbers)
    
    green = image[:, :, band_numbers[0]].astype(float)
    red = image[:, :, band_numbers[1]].astype(float)
    nir = image[:, :, band_numbers[2]].astype(float)
    swir1 = image[:, :, band_numbers[3]].astype(float)
    swir2 = image[:, :, band_numbers[4]].astype(float)
    
    # Modified Normalized Difference Water Index (MNDWI)
    mndwi = (green - swir1) / (green + swir1 + 1)
    
    return mndwi


def pseudo_subsurface_depth(image: np.ndarray, 
                           bands: Optional[List[str]] = None,
                           band_mapping: Optional[Dict] = None,
                           m1: float = 155.86,
                           m0: float = 146.46) -> np.ndarray:
    """
    Computes pseudo subsurface depth for bathymetry.

    Parameters:
    -----------
    image : np.ndarray
        The input image array with shape (height, width, num_bands)
    bands : List[str], optional
        The band codes to use. Default is ['B02', 'B03']
    band_mapping : Dict, optional
        Band mapping dictionary. Uses Sentinel-2 mapping if None.
    m1 : float, default=155.86
        Calibration parameter for blue band
    m0 : float, default=146.46
        Calibration parameter for green band
    
    Returns:
    --------
    np.ndarray
        The computed pseudo subsurface depth
    """
    if bands is None:
        bands = ['B02', 'B03']
    if band_mapping is None:
        band_mapping = DEFAULT_SENTINEL2_MAPPING
        
    band_numbers = get_band_numbers(bands, band_mapping)
    validate_band_shape(image, band_numbers)
    
    blue = image[:, :, band_numbers[0]].astype(float)
    green = image[:, :, band_numbers[1]].astype(float)
    
    # Avoid log of zero or negative values
    blue_safe = np.where(blue <= 0, 1e-6, blue)
    green_safe = np.where(green <= 0, 1e-6, green)
    
    # Pseudo-subsurface depth calculation
    depth = m1 * np.log(blue_safe) - m0 * np.log(green_safe)
    
    return depth


def mask_land(image: np.ndarray, 
             bands: Optional[List[str]] = None,
             band_mapping: Optional[Dict] = None,
             threshold: float = 0.2) -> np.ndarray:
    """
    Creates a land mask using water indices.

    Parameters:
    -----------
    image : np.ndarray
        The input image array with shape (height, width, num_bands)
    bands : List[str], optional
        The band codes to use. Default is ['B03', 'B08']
    band_mapping : Dict, optional
        Band mapping dictionary. Uses Sentinel-2 mapping if None.
    threshold : float, default=0.2
        Threshold for water detection
    
    Returns:
    --------
    np.ndarray
        Boolean mask where True indicates land
    """
    if bands is None:
        bands = ['B03', 'B08']
    if band_mapping is None:
        band_mapping = DEFAULT_SENTINEL2_MAPPING
        
    band_numbers = get_band_numbers(bands, band_mapping)
    validate_band_shape(image, band_numbers)
    
    green = image[:, :, band_numbers[0]].astype(float)
    nir = image[:, :, band_numbers[1]].astype(float)
    
    # NDWI calculation
    ndwi = (green - nir) / (green + nir + 1)  # Avoid division by zero
    
    # Land mask (NDWI < threshold indicates land)
    land_mask = ndwi < threshold
    
    return land_mask


# Convenience aliases for backward compatibility
ci = chlorophyll_index
oci = ocean_color_index
cl_oci = chlorophyll_ocean_color_index
ssi = suspended_sediment_index
ti = turbidity_index
wqi = water_quality_index
ndci = normalized_difference_chlorophyll_index
bgr = blue_green_ratio
calculate_water_surface_index = water_surface_index
calculate_pseudo_subsurface_depth = pseudo_subsurface_depth