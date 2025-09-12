import numpy as np
import geopandas as gpd
from pathlib import Path

from rasterio.warp import transform_bounds
from rasterio.coords import BoundingBox


def get_reef_gdf():
    # Load WCMC shapefile
    shapefile_path = "/home/zba21/Documents/ShallowLearn/Data/14_001_WCMC008_CoralReefs2018_v4_1/01_Data/WCMC008_CoralReef2018_Py_v4_1.shp"
    wcmc = gpd.read_file(shapefile_path)
    bbox = BoundingBox(300000.0, 8290240.0, 409800.0, 8400040.0)

    # Transform bounds
    limits = transform_bounds(32755, 'EPSG:4326', *bbox)
    subset_limits = ((limits[1], limits[0]), (limits[3], limits[2]))
    top_left = subset_limits[0]
    bottom_right = subset_limits[1]

    # Subset WCMC data
    subset = wcmc.cx[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    subset = subset.to_crs(32755)
    subset = subset[(subset.Area > 10_000) & (subset.Area < 30_000)]

    return subset