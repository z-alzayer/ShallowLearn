import pyproj
from shapely.geometry import Polygon
from shapely.ops import transform
from functools import partial

import ShallowLearn.ImageHelper as ih

def compare_extents(img_path1: str, img_path2: str) -> bool:
    """
    Compare the extents of two satellite images.
    
    Parameters:
    img_path1 (str): Path to the first satellite image.
    img_path2 (str): Path to the second satellite image.
    
    Returns:
    bool: True if the extents are the same, False otherwise.
    """
    
    # Load the first image and get its extent
    _, _, extent1 = ih.load_img(img_path1, return_meta=True)
    
    # Load the second image and get its extent
    _, _, extent2 = ih.load_img(img_path2, return_meta=True)

    
    # Compare the extents
    return extent1 == extent2



def compare_extents_with_transformation(img_path1: str, img_path2: str, target_crs: str) -> bool:
    """
    Compare the extents of two satellite images after transforming them to a common CRS.
    
    Parameters:
    img_path1 (str): Path to the first satellite image.
    img_path2 (str): Path to the second satellite image.
    target_crs (str): The target CRS to which both extents will be transformed.
    
    Returns:
    bool: True if the transformed extents are the same, False otherwise.
    """
    
    # Load the first image and get its extent and metadata
    _, meta1, extent1 = ih.load_img(img_path1, return_meta=True)
    
    # Load the second image and get its extent and metadata
    _, meta2, extent2 = ih.load_img(img_path2, return_meta=True)
    
    # Define a function to transform the extents to the target CRS
    def transform_extent(extent, src_crs, dst_crs):
        proj_transform = partial(
            pyproj.transform,
            pyproj.Proj(src_crs),
            pyproj.Proj(dst_crs))
        
        polygon = Polygon([
            (extent.left, extent.bottom),
            (extent.left, extent.top),
            (extent.right, extent.top),
            (extent.right, extent.bottom)])
        
        return transform(proj_transform, polygon)
    
    # Transform the extents to the target CRS
    transformed_extent1 = transform_extent(extent1, meta1['crs'], target_crs)
    transformed_extent2 = transform_extent(extent2, meta2['crs'], target_crs)

    print("Original extents:")
    print(extent1)
    print(extent2)
    print("Transformed extents:")
    print(transformed_extent1)
    print(transformed_extent2)

    # Compare the transformed extents
    return transformed_extent1.equals(transformed_extent2)