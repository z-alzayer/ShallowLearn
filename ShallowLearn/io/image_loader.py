"""
High-level image loading API that auto-detects file formats and applies appropriate transformations.
Replacement for ImageHelper.load_img with enhanced functionality.
"""

import numpy as np
import re
from pathlib import Path
from typing import Optional, Tuple, Union

from .satellite_data import GeoTIFFImage, LandsatImage, Sentinel2Image
from ..core.array_utils import clip_array


def load_image(
    path: Union[str, Path], 
    return_meta: bool = False, 
    clip: bool = True,
    orient: bool = True,
    file_format: Optional[str] = None
) -> Union[np.ndarray, Tuple[np.ndarray, dict, object]]:
    """
    High-level image loading function with auto-detection and proper orientation.
    
    This function serves as a replacement for ImageHelper.load_img with enhanced
    capabilities for handling different satellite data formats and file types.
    
    Parameters:
    -----------
    path : str or Path
        Path to the image file
    return_meta : bool, default False
        Whether to return metadata and bounds along with the image
    clip : bool, default True
        Whether to clip values to 0-10000 range
    orient : bool, default True
        Whether to reorient the image from (bands, height, width) to (height, width, bands)
    file_format : str, optional
        Force specific format handling ('geotiff', 'sentinel2', 'landsat')
        If None, format is auto-detected
        
    Returns:
    --------
    np.ndarray or tuple
        If return_meta=False: Image array with shape (height, width, bands) if orient=True
        If return_meta=True: Tuple of (image, metadata, bounds)
        
    Raises:
    -------
    FileNotFoundError
        If the specified file does not exist
    ValueError
        If the file format is not supported or auto-detection fails
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    
    # Auto-detect format if not specified
    if file_format is None:
        file_format = _detect_file_format(path)
    
    # Load using appropriate loader
    if file_format == 'sentinel2':
        loader = Sentinel2Image(str(path))
        img = loader.image
        if img is None:
            img = loader._load_image()  # Ensure image is loaded
    elif file_format == 'landsat':
        loader = LandsatImage(str(path))  
        img = loader.image
        if img is None:
            img = loader._load_image()  # Ensure image is loaded
    else:
        # Default to GeoTIFF loader
        loader = GeoTIFFImage(str(path))
        img = loader.load()
    
    # Apply legacy transformations for backwards compatibility
    img = _apply_legacy_transformations(img, path)
    
    # Orient image if requested (bands, height, width) -> (height, width, bands)
    if orient and len(img.shape) == 3:
        img = np.transpose(img, (1, 2, 0))
    
    # Apply clipping if requested
    if clip:
        img = clip_array(img)
    
    # Return with metadata if requested
    if return_meta:
        metadata = loader.get_metadata() if hasattr(loader, 'get_metadata') else {}
        bounds = loader.get_bounds() if hasattr(loader, 'get_bounds') else None
        return img, metadata, bounds
    
    return img


def _detect_file_format(path: Path) -> str:
    """
    Auto-detect file format based on filename patterns and content.
    
    Parameters:
    -----------
    path : Path
        Path to the image file
        
    Returns:
    --------
    str
        Detected format: 'sentinel2', 'landsat', or 'geotiff'
    """
    filename = path.name.upper()
    
    # Check for Sentinel-2 patterns
    if any(pattern in filename for pattern in ['S2A_', 'S2B_', 'MSIL1C', 'MSIL2A']):
        return 'sentinel2'
    
    # Check for Landsat patterns  
    if any(pattern in filename for pattern in ['LC08', 'LC09', 'LE07', 'LT05', 'LT04']):
        return 'landsat'
    
    # Check file extension
    if path.suffix.lower() in ['.tif', '.tiff']:
        return 'geotiff'
    
    # Default to geotiff for unknown formats
    return 'geotiff'


def _apply_legacy_transformations(img: np.ndarray, path: Path) -> np.ndarray:
    """
    Apply legacy transformations for backwards compatibility.
    
    This handles Sentinel-2 processing baseline corrections where N0400 and above
    require a -1000 offset correction.
    
    Parameters:
    -----------
    img : np.ndarray
        Input image array
    path : Path
        File path (used for specific transformations)
        
    Returns:
    --------
    np.ndarray
        Transformed image array
    """
    filename = str(path)
    
    # Apply N0400+ offset correction for Sentinel-2 processing baseline
    # Pattern matches N0400, N0401, N0500, N0511, etc. (N0400 and above)
    if re.search(r'N0[4-9][0-9][0-9]', filename):
        img = img - 1000
        
    return img


def load_image_collection(
    directory: Union[str, Path],
    pattern: str = "*.tif",
    **load_kwargs
) -> list:
    """
    Load multiple images from a directory.
    
    Parameters:
    -----------
    directory : str or Path
        Directory containing image files
    pattern : str, default "*.tif"
        Glob pattern to match files
    **load_kwargs
        Additional arguments passed to load_image()
        
    Returns:
    --------
    list
        List of loaded image arrays
    """
    directory = Path(directory)
    files = sorted(directory.glob(pattern))
    
    images = []
    for file_path in files:
        try:
            img = load_image(file_path, **load_kwargs)
            images.append(img)
        except Exception as e:
            print(f"Warning: Failed to load {file_path}: {e}")
            continue
    
    return images


# Backwards compatibility alias
load_img = load_image