import os
import geopandas as gpd
import rasterio as rio
from rasterio.mask import mask 
from shapely.geometry import box
from shapely.ops import transform
from rasterio.warp import transform_bounds
import pyproj
import matplotlib.pyplot as plt

class Clipper():
    def __init__(self, shapefile, tiff, crs=32755, prj=None, limits = None):
        self.crs = crs
        self.limits = limits
        if prj:
            self.prj_file = prj
            rio.crs.CRS.from_string(self.prj_file, morph_from_esri_dialect=True)
        
        self.shapefile = gpd.read_file(shapefile)
        self.tiff = self.read_file(tiff)
        self.subset = self.preproc_shape(self.limits)

    def preproc_shape(self, limits=None):
        """function to reduce the shapes within a shapefile using a given limit"""
        shapefile_crs = self.shapefile.crs.to_epsg()  # Dynamically get the CRS from the shapefile
        raster_crs = self.tiff.crs.to_epsg()

        if limits is None:
            # Getting bounds from raster and transforming them to the shapefile's CRS
            limits = transform_bounds(self.tiff.crs, f'EPSG:{shapefile_crs}', *self.tiff.bounds)
            # Ordering coordinates to create limits
            limits = ((limits[1], limits[0]), (limits[3], limits[2]))  # Adjusted the ordering here
        print(limits)
        top_left = limits[0]
        bottom_right = limits[1]
        subset = self.shapefile.cx[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
        subset = subset.to_crs(raster_crs)  # Transform subset CRS to match raster CRS
        subset = subset.reset_index(drop=True)        
        print(subset)
        return subset
    
    def read_file(self, tiff):
        img_file = rio.open(tiff)
        self.profile = img_file.meta.copy()
        return img_file

    def single_clip_bbox(self, shape_index, fname):
        """Clip a bounding box as opposed to single clip"""
        # 1. Extract bounding box
        bounds = self.subset.geometry[shape_index].bounds
        # 2. Create a rectangular polygon from the bounding box
        bbox_polygon = gpd.GeoSeries([box(*bounds)], crs=self.subset.crs).values[0]



        # Mask with the bounding box
        clipped, transform = mask(self.tiff, [bbox_polygon], crop=True)
        print(clipped.shape)

        clipped_meta = self.tiff.meta.copy()
        clipped_meta.update({'transform': transform, 'height': clipped.shape[1], 'width': clipped.shape[2]})
        
        with rio.open(fname, 'w', **clipped_meta) as dst:
            dst.write(clipped)


    def single_clip_exact(self, shape_index, fname):
        poly = self.subset.geometry.values[shape_index]
        clipped, transform = mask(self.tiff, [poly], crop=True)
        
        clipped_meta = self.tiff.meta.copy()
        clipped_meta.update({'transform': transform, 'height': clipped.shape[1], 'width': clipped.shape[2]})
        
        with rio.open(fname, 'w', **clipped_meta) as dst:
            dst.write(clipped)

    def multi_clip(self, fdir_out, file_name):
        """Clip all the shapes in the shapefile with a bbox
        and save them to the specified directory"""
        for keys, items in self.subset.iterrows():


            fname = os.path.join(fdir_out, f"{keys}_{file_name}")
            self.single_clip_bbox(keys, fname)


def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def get_files_with_extension(directory_path, extension):
    return [file for file in os.listdir(directory_path) if file.endswith(extension)]

limits  = ((-14.4637,145.1483),(-15.4559,146.1532))

if __name__ == "__main__":
    shape_path = "Data/14_001_WCMC008_CoralReefs2018_v4_1/01_Data/WCMC008_CoralReef2018_Py_v4_1.shp"
    output_dir = "/mnt/sda_mount/Clipped/allan/"
    raw_files = "/home/zba21/Documents/ShallowLearn/Data/"
    files = get_files_with_extension(raw_files, ".tif")
    files = [i for i in files if 'reprojected' in i]

    # files = [i for i in files if 'udm2' not in i]
    for file in files:
        create_directory_if_not_exists(os.path.join(output_dir, file[:-4]))
        output_directory = os.path.join(output_dir, file[:-4])
        print(output_directory)        
        clip = Clipper(shape_path, os.path.join(raw_files, file), limits = limits)
        clip.multi_clip(output_directory, file)
