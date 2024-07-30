import numpy as np
import geopandas as gpd
from ShallowLearn import LoadData as load_data
from rasterio.warp import transform_bounds

def get_reef_gdf(path, shapefile_path, file_path):
    # Load WCMC shapefile
    wcmc = gpd.read_file(shapefile_path)

    # Load Sentinel-2 L1C data
    data_loader = load_data.LoadSentinel2L1C(file_path)
    data_loader.load()

    # Transform bounds
    limits = transform_bounds(32755, 'EPSG:4326', *data_loader.bounds)
    subset_limits = ((limits[1], limits[0]), (limits[3], limits[2]))
    top_left = subset_limits[0]
    bottom_right = subset_limits[1]

    # Subset WCMC data
    subset = wcmc.cx[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    subset = subset.to_crs(32755)

    return subset

# Usage example
if __name__ == "__main__":
    path = "/mnt/sda_mount/All_L1C_55LCD/"
    shapefile_path = "/home/zba21/Documents/ShallowLearn/Data/14_001_WCMC008_CoralReefs2018_v4_1/01_Data/WCMC008_CoralReef2018_Py_v4_1.shp"
    file_path = "/mnt/sda_mount/All_L1C_55LCD/S2B_MSIL1C_20220406T003659_N0400_R059_T55LCD_20220406T015555.SAFE/MTD_MSIL1C.xml"
    reef_gdf = get_reef_gdf(path, shapefile_path, file_path)
