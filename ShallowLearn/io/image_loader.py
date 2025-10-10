"""
High-level image loading API that auto-detects file formats and applies appropriate transformations.
Replacement for ImageHelper.load_img with enhanced functionality.
"""

import re
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import rasterio as rio
import rasterio.mask
from shapely.geometry import box
from shapely.ops import unary_union

from ..core.array_utils import clip_array
from .satellite_data import GeoTIFFImage, LandsatImage, Sentinel2Image, SpotImage


def load_image(
    path: Union[str, Path],
    return_meta: bool = False,
    clip: bool = False,
    file_format: Optional[str] = None,
    gdf_clip: Optional[object] = None,
    return_sat_object = False,
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
    clip : bool, default False
        Whether to clip values to 0-10000 range
    file_format : str, optional
        Force specific format handling ('geotiff', 'sentinel2', 'landsat', 'spot')
        If None, format is auto-detected
    gdf_clip : GeoDataFrame, optional
        GeoDataFrame with geometries for clipping the image. If provided, the image will be clipped to the geometries
    return_sat_object: False
        Returns custom SatelliteImage with additional methods

    Returns:
    --------
    np.ndarray or tuple
        If return_meta=False: Image array with shape (height, width, bands)
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
    if file_format == "sentinel2":
        loader = Sentinel2Image(str(path))
        img = loader.image
        if img is None:
            img = loader._load_image()  # Ensure image is loaded
    elif file_format == "landsat":
        loader = LandsatImage(str(path))
        img = loader.image
        if img is None:
            img = loader._load_image()  # Ensure image is loaded
    elif file_format == "spot":
        loader = SpotImage(str(path))
        img = loader.image
        if img is None:
            img = loader._load_image()  # Ensure image is loaded
    else:
        # Default to GeoTIFF loader
        loader = GeoTIFFImage(str(path))
        img = loader.load()

    # TODO: Tidy this up so its behaviourally consistent and not an additional state to manage
    if return_sat_object:
        return loader
    # Apply legacy transformations for backwards compatibility
    img = _apply_legacy_transformations(img, path)

    # Always ensure channels-last format (height, width, bands)
    if len(img.shape) == 3:
        # Check if image needs to be transposed from (bands, height, width) to (height, width, bands)
        if img.shape[0] <= 20 and img.shape[2] > 20:  # Likely (bands, height, width)
            img = np.transpose(img, (1, 2, 0))
        # If it's already (height, width, bands) or we can't determine, leave as is

    # Apply GeoDataFrame clipping if requested
    if gdf_clip is not None:
        metadata = loader.get_metadata() if hasattr(loader, "get_metadata") else {}
        img = clip_image_with_gdf(img, gdf_clip, metadata.get('crs'), metadata.get('transform'))

    # Apply value clipping if requested
    if clip:
        img = clip_array(img)

    # Return with metadata if requested
    if return_meta:
        metadata = loader.get_metadata() if hasattr(loader, "get_metadata") else {}
        bounds = loader.get_bounds() if hasattr(loader, "get_bounds") else None
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
        Detected format: 'sentinel2', 'landsat', 'spot', or 'geotiff'
    """
    filename = path.name.upper()

    # Check for Sentinel-2 patterns first (highest priority for satellite data)
    if any(pattern in filename for pattern in ["S2A_", "S2B_", "MSIL1C", "MSIL2A"]):
        return "sentinel2"

    # Check for Landsat patterns
    if any(pattern in filename for pattern in ["LC08", "LC09", "LE07", "LT05", "LT04"]):
        return "landsat"

    # Check for SPOT patterns
    # SPOT filename pattern examples:
    # - 001-002_S5_173-309-8_2002-07-26-06-52-08_HRG-2_J_MX_KK.vrt
    # - Contains satellite identifier like S1, S2, S4, S5
    # - Contains instrument like HRV, HRVIR, HRG, HRS
    if any(pattern in filename for pattern in ["_S1_", "_S2_", "_S4_", "_S5_"]):
        return "spot"

    # Also check for SPOT instrument names in VRT files
    if path.suffix.lower() == ".vrt":
        if any(pattern in filename for pattern in ["HRV", "HRVIR", "HRG", "HRS"]):
            # Need to verify it's SPOT and not something else
            # Check if it has SPOT satellite identifier pattern
            if re.search(r"_S[1-5]_", filename):
                return "spot"

    # Check file extension for standard geotiff files (after satellite pattern matching)
    if path.suffix.lower() in [".tif", ".tiff"]:
        return "geotiff"

    # For VRT files without clear satellite patterns, try to peek at metadata
    if path.suffix.lower() == ".vrt":
        try:
            with rio.open(path) as src:
                # Check for SPOT tags in the VRT
                spot_tags = src.tags(ns="SPOT")
                if spot_tags and "SATELLITE" in spot_tags:
                    return "spot"

                # Check band descriptions for SPOT-like patterns (XS1, XS2, etc.)
                if src.descriptions and any(
                    desc and re.match(r"^XS\d+$|^SWIR$|^PAN$|^NIR$", desc)
                    for desc in src.descriptions
                ):
                    return "spot"
        except Exception:
            pass  # If we can't read it, fall back to geotiff

    # Default to geotiff for unknown formats
    return "geotiff"


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
    if re.search(r"N0[4-9][0-9][0-9]", filename):
        img = img - 1000

    return img


def load_image_collection(
    directory: Union[str, Path], pattern: str = "*.tif", **load_kwargs
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


def clip_image_with_gdf(
    img: np.ndarray, 
    gdf: Union[object, List[object]], 
    crs: object, 
    transform: object
) -> np.ndarray:
    """
    Clip image using GeoDataFrame(s) by creating an in-memory raster.
    
    This function handles complex file formats (like Sentinel-2 ZIP files) by using
    the provided CRS and transform information instead of trying to reopen files.
    
    Parameters:
    -----------
    img : np.ndarray
        Input image array in channels-last format (height, width, bands)
    gdf : GeoDataFrame or List[GeoDataFrame]
        Single GeoDataFrame or list of GeoDataFrames with geometries for clipping.
        The function will clip by the bounding box of all geometries combined.
    crs : rasterio.crs.CRS or similar
        Coordinate reference system of the image
    transform : rasterio.Affine
        Affine transform of the image
    
    Returns:
    --------
    np.ndarray
        Clipped image array in channels-last format
        
    Notes:
    ------
    - For multiple GDFs, clips by the combined bounding box of all geometries
    - Handles CRS conversion automatically
    - Returns original image if clipping fails
    """
    if crs is None or transform is None:
        print("Warning: Missing CRS or transform information for GDF clipping")
        print("Returning original image without clipping")
        return img
    
    # Handle single GDF or list of GDFs
    if not isinstance(gdf, list):
        gdf_list = [gdf]
    else:
        gdf_list = gdf
    
    # Collect all geometries from all GDFs
    all_geometries = []
    for single_gdf in gdf_list:
        # Convert GDF to same CRS as image if needed
        if single_gdf.crs != crs:
            single_gdf_projected = single_gdf.to_crs(crs)
        else:
            single_gdf_projected = single_gdf
        
        # Add all geometries from this GDF
        all_geometries.extend([geom for geom in single_gdf_projected.geometry])
    
    if not all_geometries:
        print("Warning: No valid geometries found in GDF(s)")
        print("Returning original image without clipping")
        return img
    
    # Create unified geometry from all geometries
    # Use unary_union to handle overlapping geometries efficiently
    unified_geom = unary_union(all_geometries)
    
    # Convert to list of shapes for rasterio.mask
    if hasattr(unified_geom, '__iter__') and not hasattr(unified_geom, 'geom_type'):
        # Multi-geometry result
        shapes = list(unified_geom)
    else:
        # Single geometry result  
        shapes = [unified_geom]
    
    print(f"Clipping with {len(shapes)} unified geometries")
    
    # Create in-memory raster from the numpy array
    # Convert from channels-last to bands-first for rasterio
    if len(img.shape) == 3:
        img_bands_first = np.transpose(img, (2, 0, 1))
        height, width, count = img.shape
    else:
        img_bands_first = img[np.newaxis, :, :]
        height, width = img.shape
        count = 1
    
    # Create temporary in-memory raster
    with rio.MemoryFile() as memfile:
        with memfile.open(
            driver='GTiff',
            height=height,
            width=width,
            count=count,
            dtype=img.dtype,
            crs=crs,
            transform=transform
        ) as dataset:
            # Write image data to the dataset
            dataset.write(img_bands_first)
            
            # Apply the mask - clip to the unified geometries
            clipped_data, clipped_transform = rio.mask.mask(
                dataset, shapes, crop=True, nodata=0
            )
            
            # Convert back to channels-last format
            if len(clipped_data.shape) == 3 and clipped_data.shape[0] <= 20:
                clipped_img = np.transpose(clipped_data, (1, 2, 0))
            else:
                clipped_img = clipped_data
            
            # Ensure output has same dtype as input
            clipped_img = clipped_img.astype(img.dtype)
            
            # Calculate and report clipping statistics
            original_pixels = img.shape[0] * img.shape[1] if len(img.shape) >= 2 else img.size
            clipped_pixels = clipped_img.shape[0] * clipped_img.shape[1] if len(clipped_img.shape) >= 2 else clipped_img.size
            
            if clipped_pixels < original_pixels:
                reduction = ((original_pixels - clipped_pixels) / original_pixels * 100)
                print(f"GDF clipping successful: {reduction:.1f}% pixel reduction")
            else:
                print("GDF clipping completed: no significant size reduction")
            
            return clipped_img


# Backwards compatibility alias
load_img = load_image

