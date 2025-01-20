
import rasterio 
import rasterio.mask
import geopandas as gpd 
import numpy as np
from scipy import linalg
from scipy import stats

def extract_raster_values(shapefile_path, raster_path):
    """
    Extract raster values using shapefile geometries as masks.
    
    Args:
        shapefile_path: Path to the shapefile
        raster_path: Path to the raster file
    
    Returns:
        List of numpy arrays containing masked raster values for each geometry
    """
    # Read the shapefile using GeoPandas
    shapes = gpd.read_file(shapefile_path)
    
    # Read the raster using rasterio
    with rasterio.open(raster_path) as src:
        # Check if CRS match
        if shapes.crs != src.crs:
            # Reproject shapefile to match raster CRS
            shapes = shapes.to_crs(src.crs)[1]
        
        # Create list to store values for each geometry
        values = []
        
        # Extract values for each geometry
        for geometry in shapes.geometry:
            # Create mask from geometry
            mask = rasterio.mask.mask(src, [geometry], crop=True)
            # Fix later - hardcoded baseline 4+ value conversion
            out_image = (mask[0] - 1000) / 10_000
            
            # Store masked values
            values.append(out_image)
            
    return values

def calculate_depth_invariant_indices(deep_areas, shallow_areas, band_i_idx, band_j_idx):
    
    # Calculate deep water means
    deep_i = np.concatenate([area[band_i_idx].flatten() for area in deep_areas])
    deep_j = np.concatenate([area[band_j_idx].flatten() for area in deep_areas])
    Ls_i = np.nanmean(deep_i)
    Ls_j = np.nanmean(deep_j)
    
    # Calculate regression for same bottom areas
    shallow_i = np.concatenate([area[band_i_idx].flatten() for area in shallow_areas])
    shallow_j = np.concatenate([area[band_j_idx].flatten() for area in shallow_areas])
    
    # Apply minimum difference threshold of 0.01
    dif_i = np.maximum(shallow_i - Ls_i, 0.01)
    dif_j = np.maximum(shallow_j - Ls_j, 0.01)
    
    # Transform to log space
    Xi = np.log(dif_i)
    Xj = np.log(dif_j)
    
    # Calculate perpendicular regression slope
    slope, _ = stats.linregress(Xi, Xj)[:2]
    
    return slope, (Ls_i, Ls_j)

def apply_depth_invariant_index(image_i, image_j, ki_kj, Ls):
    """
    Apply the depth invariant index to full images
    """
    Ls_i, Ls_j = Ls
    dif_i = np.maximum(image_i - Ls_i, 0.01)
    dif_j = np.maximum(image_j - Ls_j, 0.01)
    return np.log(dif_i) - ki_kj * np.log(dif_j)

