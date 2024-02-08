import numpy as np
from ShallowLearn.band_mapping import band_mapping
from ShallowLearn.PreprocDecorators import remove_zeros_decorator
# Define constants and parameters
MNDWI_thr = 0.2
NDWI_thr = 0.2
filter_UABS = True
filter_SSI = False
SDBgreen = True
cs = 0
preAnalysis = False
m1 = 155.86
m0 = 146.46
mp = 1000
pSDBmin = 0.201
pSDBmax = 4.983
nConst = 1000





@remove_zeros_decorator
def ci(image, bands=None):
    """
    Computes the Chlorophyll Index (CI) using the specified band codes.
    
    Purpose:
    The Chlorophyll Index is used to estimate chlorophyll content in vegetation. It provides information about the health and vigor of plants.

    Reference: https://www.tandfonline.com/doi/pdf/10.1080/0143116042000274015?casa_token=L7WoAXpR_SkAAAAA:-Af2eztPG_JLHT66_adHNhj4zs9tX6rXZys2jVIjet0ZdWFCAsXz4s-Eq2ZOnVWBOAfLrcdVJT_-yQ
    
    Parameters:
    - image: numpy array, shape (height, width, num_bands)
      The input image array.
    - bands: list of str, optional
      The band codes to use for computing the index. Default is ['B04', 'B05', 'B06', 'B07']
    
    Returns:
    - ci: numpy array, shape (height, width)
      The computed Chlorophyll Index.
    """
    if bands is None:
        bands = ['B04', 'B05', 'B06', 'B07']  # Default band codes
    band_numbers = get_band_numbers(bands)
    validate_band_shape(image, band_numbers)
    r665 = image[:, :, band_numbers[0]]
    r705 = image[:, :, band_numbers[1]]
    r740 = image[:, :, band_numbers[2]]
    r783 = image[:, :, band_numbers[3]]

    ri = r665 + r783

    rep = 700 + 40 * (ri - r705) / ((r740 + r705) + 1) # Avoid division by zero
    return rep

@remove_zeros_decorator
def oci(image, bands=None):
    """
    Computes the Ocean Color Index (OCI) using the specified band codes.
    
    Purpose:
    The Ocean Color Index is used to assess ocean color properties, particularly the presence of chlorophyll. It helps in studying phytoplankton abundance and water quality in marine environments.
    
    reference: https://www.sciencedirect.com/science/article/pii/S0034425709001710?casa_token=b3lUx1eEEpcAAAAA:xD37DQIjMsdznvPETd0Ex9oWHid0XsQwXyt6B_N5E3pd-tk145ffxv3RqY2FtelBqk0ubYuAPAg

    Parameters:
    - image: numpy array, shape (height, width, num_bands)
      The input image array.
    - bands: list of str, optional
      The band codes to use for computing the index. Default is ['B03', 'B12'].
    
    Returns:
    - oci: numpy array, shape (height, width)
      The computed Ocean Color Index.
    """
    if bands is None:
        bands = ['B04','B08','B12']  # Default band codes
    band_numbers = get_band_numbers(bands)
    validate_band_shape(image, band_numbers)
    red = image[:, :, band_numbers[0]]
    nir = image[:, :, band_numbers[1]]
    swir = image[:, :, band_numbers[2]]
    fai = red + (swir - red) * (nir - red) / ((swir - red) + 1) # Avoid division by zero
    # Avoid division by zero
    # swir_band[swir_band == 0] = np.nan
    
    # oci = (green_band / swir_band) - 1
    return fai

@remove_zeros_decorator
def cl_oci(image, bands=None):
    """
    Computes the Ocean Color Index (OCI) using the specified band codes.
    
    Purpose:
    The Ocean Color Index is used to assess ocean color properties, particularly the presence of chlorophyll. It helps in studying phytoplankton abundance and water quality in marine environments.
    
    reference: https://agupubs.onlinelibrary.wiley.com/doi/pdfdirect/10.1029/2019JC015498

    Parameters:
    - image: numpy array, shape (height, width, num_bands)
      The input image array.
    - bands: list of str, optional
      The band codes to use for computing the index. Default is ['B03', 'B12'].
    
    Returns:
    - oci: numpy array, shape (height, width)
      The computed Ocean Color Index.
    """
    if bands is None:
        bands = ['B02','B03','B04']  # Default band codes
    band_numbers = get_band_numbers(bands)
    validate_band_shape(image, band_numbers)
    blue = image[:, :, band_numbers[0]]
    green = image[:, :, band_numbers[1]]
    red = image[:, :, band_numbers[2]]
    cl_oci = green - ((blue) + (green - blue)) / (red - blue) * (red - blue) + 1
    return cl_oci

@remove_zeros_decorator
def ssi(image, bands=None):
    """
    Computes the Suspended Sediment Index (SSI) using the specified band codes.
    
    Purpose:
    The Suspended Sediment Index is used to estimate the concentration of suspended sediments in water bodies. It helps in monitoring water quality, sediment transport, and erosion processes.
    
    Parameters:
    - image: numpy array, shape (height, width, num_bands)
      The input image array.
    - bands: list of str, optional
      The band codes to use for computing the index. Default is ['B04', 'B05', 'B08'].
    
    Returns:
    - ssi: numpy array, shape (height, width)
      The computed Suspended Sediment Index.
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

@remove_zeros_decorator
def ti(image, bands=None):
    """
    Computes the Turbidity Index (TI) using the specified band codes.
    
    Purpose:
    The Turbidity Index is used to estimate the turbidity in water bodies. It helps in monitoring water quality, sediment transport, and erosion processes.
    
    Parameters:
    - image: numpy array, shape (height, width, num_bands)
      The input image array.
    - bands: list of str, optional
      The band codes to use for computing the index. Default is ['B04', 'B08'].
    
    Returns:
    - ti: numpy array, shape (height, width)
      The computed Turbidity Index.
    """
    if bands is None:
        bands = ['B04', 'B08']  # Default band codes
    band_numbers = get_band_numbers(bands)
    validate_band_shape(image, band_numbers)
    red_band = image[:, :, band_numbers[0]]
    nir_band = image[:, :, band_numbers[1]]
    
    # Check for divide by zero errors
    denominator = nir_band + red_band
    denominator = denominator.astype(float)
    denominator[denominator == 0] = np.nan  # Replace zero with NaN to avoid divide by zero errors
    
    ti = (nir_band - red_band) / (denominator + 1)
    
    # Check for NaN values in the result
    ti = np.nan_to_num(ti, nan=0.0)  # Replace NaN values with 0.0
    
    return ti
@remove_zeros_decorator
def wqi(image, bands=None):
    """
    Computes the Water Quality Index (WQI) using the specified band codes.
    
    Purpose:
    The Water Quality Index is used to assess water quality based on multiple parameters. It provides a comprehensive measure of water health, considering the contributions of various bands to the index computation.
    
    Parameters:
    - image: numpy array, shape (height, width, num_bands)
      The input image array.
    - bands: list of str, optional
      The band codes to use for computing the index. Default is ['B03', 'B04', 'B05', 'B06'].
    
    Returns:
    - wqi: numpy array, shape (height, width)
      The computed Water Quality Index.
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

@remove_zeros_decorator
def ndci(image, bands=None):
    """
    Computes the Normalized Difference Chlorophyll Index (NDCI) using the specified band codes.
    
    Purpose:
    The Normalized Difference Chlorophyll Index is used to estimate chlorophyll content in vegetation. It provides a normalized measure of the difference between green reflectance and red-edge reflectance, indicating vegetation health.
    
    Parameters:
    - image: numpy array, shape (height, width, num_bands)
      The input image array.
    - bands: list of str, optional
      The band codes to use for computing the index. Default is ['B03', 'B05'].
    
    Returns:
    - ndci: numpy array, shape (height, width)
      The computed Normalized Difference Chlorophyll Index.
    """
    if bands is None:
        bands = ['B03', 'B05']  # Default band codes
    band_numbers = get_band_numbers(bands)
    validate_band_shape(image, band_numbers)
    green_band = image[:, :, band_numbers[0]]
    red_edge_band = image[:, :, band_numbers[1]]
    ndci = (green_band - red_edge_band) / (green_band + red_edge_band) + 1
    return ndci

@remove_zeros_decorator
def cloud_index(image, bands=None):
    """
    Computes the Cloud Index using the specified band codes.
    
    Purpose:
    The Cloud Index is used to identify and estimate the presence of clouds in satellite images. High values typically indicate the presence of clouds.
    
    Parameters:
    - image: numpy array, shape (height, width, num_bands)
      The input image array.
    - bands: list of str, optional
      The band codes to use for computing the index. Default is ['B08', 'B11'].
    
    Returns:
    - ci: numpy array, shape (height, width)
      The computed Cloud Index.
    """
    if bands is None:
        bands = ['B08', 'B11']  # Default band codes for NIR and SWIR
    band_numbers = get_band_numbers(bands)
    validate_band_shape(image, band_numbers)
    nir_band = image[:, :, band_numbers[0]]
    swir_band = image[:, :, band_numbers[1]]
    ci = swir_band / nir_band
    return ci

# def wbei(image, bands=None):
#     """
#     Computes the Water Body Extraction Index (WBEI) using the specified band codes.
    
#     Purpose:
#     The Water Body Extraction Index is used to extract water bodies from remote sensing imagery. It helps in distinguishing water pixels from other land features based on the spectral properties of water bodies.
    
#     Parameters:
#     - image: numpy array, shape (height, width, num_bands)
#       The input image array.
#     - bands: list of str, optional
#       The band codes to use for computing the index. Default is ['B04', 'B03', 'B02'].
    
#     Returns:
#     - wbei: numpy array, shape (height, width)
#       The computed Water Body Extraction Index.
#     """
#     if bands is None:
#         bands = ['B04', 'B03', 'B02']  # Default band codes
#     band_numbers = get_band_numbers(bands)
#     validate_band_shape(image, band_numbers)
#     red_band = image[:, :, band_numbers[0]]
#     green_band = image[:, :, band_numbers[1]]
#     blue_band = image[:, :, band_numbers[2]]
#     wbei = (green_band - red_band) / (green_band + red_band + blue_band)
#     return wbei
@remove_zeros_decorator
def bgr(image, bands=None):
    """
    Computes the Blue to Green Ratio (BGR) using the specified band codes.
    
    Purpose:
    The Blue to Green Ratio is used to assess water quality by comparing the blue and green reflectance values. It provides information about the concentration of chlorophyll and suspended sediments in water bodies.
    
    Parameters:
    - image: numpy array, shape (height, width, num_bands)
      The input image array.
    - bands: list of str, optional
      The band codes to use for computing the index. Default is ['B02', 'B03'].
    
    Returns:
    - bgr: numpy array, shape (height, width)
      The computed Blue to Green Ratio.
    """
    if bands is None:
        bands = ['B02', 'B03']  # Default band codes
    band_numbers = get_band_numbers(bands)
    validate_band_shape(image, band_numbers)
    blue_band = image[:, :, band_numbers[0]]
    green_band = image[:, :, band_numbers[1]]
    
    # Avoid division by zero
    green_band = green_band.astype(float)
    green_band[green_band == 0] = np.nan
    
    bgr = blue_band / green_band
    return bgr
def mask_land(image, land_band='B11', threshold=10):
    """
    Masks out the land areas in an image using the specified land band.
    
    Purpose:
    This function is used to mask out the land areas in a satellite image. It uses a specified band (default is Band 11) and a threshold value to determine which areas are land.
    
    Parameters:
    - image: numpy array, shape (height, width, num_bands)
      The input image array.
    - land_band: a string, optional
      The band code to use for identifying land. Default is 'B11'.
    - threshold: float, optional
      The threshold value to use for identifying land. Default is 10.
    
    Returns:
    - masked_image: numpy array, shape (height, width, num_bands)
      The input image array with land areas masked out.
    """
    land_band = [land_band]
    band_number = get_band_numbers(land_band)
    #validate_band_shape(image, [band_number])
    land_band_image = image[:, :, band_number]
    
    # Create a mask where land areas (values above the threshold) are True
    land_mask = land_band_image > threshold
    
    
    return land_mask


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

@remove_zeros_decorator
def calculate_water_surface_index(band_array, bands = None):
    """
    Calculates Water Surface Index based on multiple spectral indices.
    
    Parameters:
    band_array (numpy.ndarray): Multiband array with all Sentinel bands
    
    Returns:
    int: Water surface index (1: water, 0: non-water)
    """
    if bands is None:
        bands = ['B03', 'B04', 'B08', 'B11', 'B12']  # Default band codes
    band_numbers = get_band_numbers(bands)
    validate_band_shape(band_array, band_numbers)

    green_band = band_array[:, :, band_numbers[0]]
    red_band = band_array[:, :, band_numbers[1]]
    nir_band = band_array[:, :, band_numbers[2]]
    swir1_band = band_array[:, :, band_numbers[3]]
    swir2_band = band_array[:, :, band_numbers[4]]

    wsi = (green_band + red_band) - (nir_band + swir1_band + swir2_band)
    
    return wsi


# Function to calculate pseudo Subsurface Depth
@remove_zeros_decorator
def calculate_pseudo_subsurface_depth(band_array, bands = None):
    """
    Calculates pseudo Subsurface Depth based on the logarithmic ratio of the blue band to the green or red band.
    
    Parameters:
    band_array (numpy.ndarray): Multiband array with all Sentinel bands
    
    Returns:
    float: Pseudo subsurface depth
    """
    if bands is None:
        bands = ['B02', 'B03']  # Default band codes
    band_numbers = get_band_numbers(bands)
    validate_band_shape(band_array, band_numbers)

    blue_band = band_array[:, :, band_numbers[0]]
    green_or_red_band = band_array[:, :, band_numbers[1]]

    pseudo_sdb = np.log(green_or_red_band / blue_band)
    return pseudo_sdb



# Function to calculate actual Subsurface Depth
@remove_zeros_decorator
def calculate_subsurface_depth(pseudo_SDB):
    """
    Calculates actual Subsurface Depth from the pseudo Subsurface Depth using a linear transformation.
    
    Parameters:
    pseudo_SDB (float): Pseudo subsurface depth
    
    Returns:
    float: Actual subsurface depth
    """
    return m1 * pseudo_SDB - m0
