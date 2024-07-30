import rasterio
import numpy as np
from rasterio.warp import reproject, Resampling
from rasterio.enums import Resampling as ResamplingEnums


def get_raster_dimensions(dataset_path):
    """
    Retrieve the width and height of a raster dataset.

    Parameters:
    - dataset_path: str, the path to the raster dataset.

    Returns:
    - tuple: (width, height) dimensions of the raster.
    """

    # Open the dataset using rasterio
    with rasterio.open(dataset_path) as dataset:
        width = dataset.width
        height = dataset.height

    return (width, height)



def resample_raster(dataset_path, band_index, target_width, target_height, resampling_method=ResamplingEnums.bilinear):
    """
    Resample a specific band in a raster dataset to a new width and height.

    Parameters:
    - dataset_path: str, the path to the raster dataset to resample.
    - band_index: int, the index of the band to resample (1-based index).
    - target_width: int, the target width of the output image.
    - target_height: int, the target height of the output image.
    - resampling_method: rasterio.enums.Resampling, the method used for resampling.

    Returns:
    - numpy.ndarray: the resampled raster data for the specified band.
    """

    # Open the original dataset
    with rasterio.open(dataset_path) as dataset:
        # Validate band_index
        if band_index < 1 or band_index > dataset.count:
            raise ValueError("The band_index is out of the allowable range.")

        # Get the original data's CRS, transform, and datatype
        data_crs = dataset.crs
        data_transform = dataset.transform
        data_dtype = dataset.dtypes[0]

        # Calculate the new transform for the target dimensions
        new_transform = rasterio.transform.from_bounds(
            *dataset.bounds, width=target_width, height=target_height)

        # Prepare an array to hold the resampled data
        resampled_data = np.zeros((target_height, target_width), dtype=data_dtype)

        # Perform the reprojection (resampling) for the specified band
        reprojected = reproject(
            source=rasterio.band(dataset, band_index),
            destination=resampled_data,
            src_transform=data_transform,
            src_crs=data_crs,
            dst_transform=new_transform,
            dst_crs=data_crs,
            resampling=resampling_method
        )

    return resampled_data



def resample_to_target(src_array, target_array, index, resampling_method = ResamplingEnums.bilinear):

    width, height = get_raster_dimensions(src_array)

    return resample_raster(target_array, index, width, height, resampling_method= resampling_method)