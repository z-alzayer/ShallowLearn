import numpy as np
import rasterio
import csv
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from rasterio.enums import Resampling
from osgeo import gdal
from rasterio.warp import reproject
from rasterio.mask import mask
from shapely.geometry import box
import zipfile
import os

from ShallowLearn.FileProcessing import list_files_in_dir_recur
from ShallowLearn.DateHelper import extract_dates, get_season, southern_hemisphere_meteorological_seasons, extract_individual_date
import ShallowLearn.ImageHelper as ih
from ShallowLearn.PreprocDecorators import remove_zeros_decorator
from ShallowLearn.Indices import calculate_water_surface_index, mask_land, cloud_index
from ShallowLearn.band_mapping import band_mapping
import ShallowLearn.ResamplingMethods as rs

class DataLoader:
    def __init__(self, data_source):
        self.data_source = data_source

    def transform(self):
        """Abstract transform method that should be implemented by the subclass"""
        raise NotImplementedError("This method should be implemented by the subclass")

    def load(self):
        """Abstract load method that should be implemented by the subclass"""
        raise NotImplementedError("This method should be implemented by the subclass")

class LoadNumpyArray(DataLoader):
    """Loads a numpy array of remotely sensed data"""
    def __init__(self, data_source):
        super().__init__(data_source)

    def load(self):
        # Implement the method to load remote sensing data using NumPy
        data = np.load(self.data_source)
        return data

class LoadGeoTIFF(DataLoader):
    """Loads a geotiff directly from a raster accepted by rasterio"""
    def __init__(self, data_source):
        super().__init__(data_source)
        self.metadata = None
        self.bounds = None
    
    def load(self):
        # Implement the method to load a GeoTIFF file using Rasterio
        with rasterio.open(self.data_source) as src:
            data = src.read()
            no_data = src.nodatavals
            # masked_data = np.ma.masked_array(data, mask=[band_data == nodata for band_data, nodata in zip(data, no_data)])
        return data

    def get_metadata(self):
        with rasterio.open(self.data_source) as src:
            self.metadata = src.meta
        return self.metadata

    
    def get_bounds(self):
        with rasterio.open(self.data_source) as src:
            self.bounds = src.bounds
        return self.bounds

class PVI_Dataloader(DataLoader):
    def __init__(self, data_source):
        self.is_zip = data_source.endswith(".zip")
        if self.is_zip:
            try:
                with zipfile.ZipFile(data_source, 'r') as zip_ref:
                    self.files = [f for f in zip_ref.namelist() if "PVI" in f ][0]
            except:
                print(f"File: {data_source} failed. Please double check integrity of file")
        
        self.zip_path = f"zip+file://{data_source}/{self.files}"
        print(self.zip_path)
    
    def load(self):
        with rasterio.open(self.zip_path) as dataset:
            self.pvi_image = dataset.read()
            self.pvi_image = np.swapaxes(self.pvi_image, 0, 2)
            self.pvi_image = np.swapaxes(self.pvi_image, 0, 1)
        return self.pvi_image
    

class LoadSentinel2L1C(DataLoader):
    def __init__(self, data_source, band_mapping=band_mapping):
        super().__init__(data_source)
        self.band_mapping = band_mapping
        self.is_zip = data_source.endswith(".zip")

        if self.is_zip:
            with zipfile.ZipFile(data_source, 'r') as zip_ref:
                self.files = [f for f in zip_ref.namelist() if "MTD_MSIL1C.xml" in f or "MTD_MSIL2A.xml" in f]
            self.zip_path = f"/vsizip/{data_source}"

        elif data_source.endswith(".xml"):
            self.files = [data_source]
        elif data_source.endswith(".SAFE"):
            self.files = [i for i in list_files_in_dir_recur(data_source) if "MTD_MSIL1C" in i]

        if len(self.files) == 0 or len(self.files) > 1:
            raise Exception("Multiple or no MTD_MSIL1C files found, please double check your data")
        
        self.file = self.files[0]
        if self.is_zip:
            self.file = os.path.join(self.zip_path, self.file)

    def load(self):
        """Load subdatasets from the primary file"""
        with rasterio.open(self.file) as dataset:
            self.subdatasets = dataset.subdatasets

        # Open the first subdataset
        with rasterio.open(self.subdatasets[0]) as first_array:
            self.tags = first_array.tags()
            self.profile = first_array.profile
            self.metadata = first_array.meta
            self.offsets = first_array.offsets
            self.bounds = first_array.bounds

        return self.subdatasets

    def get_resolution_subdatasets(self):
        """Retrieve subdatasets based on resolution categories"""
        subdatasets = self.load()
        resolutions = {
            '10m': [s for s in subdatasets if "10m" in s],
            '20m': [s for s in subdatasets if "20m" in s],
            '60m': [s for s in subdatasets if "60m" in s],
            'tci': [s for s in subdatasets if "TCI" in s]
        }
        return resolutions
    def describe_bands(self):
        """
        Describes the bands in the datasets, identifying their purpose based on the filename and metadata.
        """
        resolution_datasets = self.get_resolution_subdatasets()
        description_dict = {}
        for key, items in resolution_datasets.items():
            if "tci" in key:
                continue
            else:
                with rasterio.open(items[0]) as ds:
                    description_list = [i for i in ds.descriptions]
                    description_dict[items[0]] = [i.split(",")[0] for i in description_list]
        return description_dict
    
    def get_selected_bands(self, resolution = "10m", selected_bands = ['B02', 'B03', 'B04','B08']):
        band_description = self.describe_bands()
        width, height = rs.get_raster_dimensions(self.get_resolution_subdatasets()[resolution][0])
        counter = len(selected_bands)
        idx_location = len(selected_bands)
        index_dictionary = {}

        for key, item in band_mapping.items():
            if key in selected_bands:
                for path, band_list in band_description.items():
                    if key[1] == "0":
                        substring = key.replace("0","")
                    else:
                        substring = key
                    if substring in band_list:
                        band_index = band_list.index(substring) + 1
                        index_dictionary[idx_location - counter] = [path, band_index, key]
                        counter -= 1
        return index_dictionary, width, height
    
    def construct_resampled_array(self, resolution = "10m", selected_bands = ['B02', 'B03', 'B04', 'B08']):
        selected_bands_dict, width, height = self.get_selected_bands(resolution, selected_bands)
        empty_array = np.zeros((width, height, len(selected_bands)))
        for keys, items in selected_bands_dict.items():
            empty_array[:,:,keys] = rs.resample_raster(items[0], items[1], width, height )
        return empty_array
    
    def clip_raster_with_shape(self, shapes, resolution='10m', selected_bands=['B02', 'B03', 'B04', 'B08'], use_mask=True):
        """Clip the raster with a shape from a shapefile and then resample, optionally using the shape's bounding box."""
        selected_bands_dict, target_width, target_height = self.get_selected_bands(resolution, selected_bands)
        clipped_and_resampled = []

        final_width, final_height = None, None
        geometry = [shapes.geometry.iloc[0]] if use_mask else [box(*shapes.geometry.iloc[0].bounds)]

        # Determine the final width and height from the highest resolution band
        for band_key, band_path in selected_bands_dict.items():
            if resolution in band_path[0]:
                print(band_path)
                with rasterio.open(band_path[0]) as src:
                    clipped_image, clipped_transform = mask(src, geometry, crop=True)
                    clipped_height, clipped_width = clipped_image.shape[1:3]
                    final_width, final_height = clipped_width, clipped_height
                    break

        if final_width is None or final_height is None:
            raise ValueError("Could not determine the final width and height for resampling.")

        # Clip and resample each band
        for band_key, band_path in selected_bands_dict.items():
            with rasterio.open(band_path[0]) as src:
                clipped_image, clipped_transform = mask(src, geometry, crop=True)
                resampled_data = np.empty((final_height, final_width), dtype=clipped_image.dtype)
                
                new_transform = rasterio.transform.from_bounds(
                    *rasterio.transform.array_bounds(clipped_image.shape[1], clipped_image.shape[2], clipped_transform),
                    width=final_width,
                    height=final_height
                )

                reprojected, _ = reproject(
                    source=clipped_image[band_path[1] - 1],
                    destination=resampled_data,
                    src_transform=clipped_transform,
                    src_crs=src.crs,
                    dst_transform=new_transform,
                    dst_crs=src.crs,
                    resampling=Resampling.bilinear
                )

                clipped_and_resampled.append(reprojected)

        return np.array(clipped_and_resampled)




class LoadFromCSV(DataLoader):
    """This class loads data from a CSV file. The CSV file should contain the file paths of the data to be loaded.
        It assumes that the file paths are in a named column of the CSV file, passed to the data loader as col_name"""
    def __init__(self, data_source):
        super().__init__(data_source)
        self.data_source = pd.read_csv(self.data_source)

    def get_specific_reef(self, reef_id):
        """This method returns the a data frame of a specific reef"""
        reef = self.data_source.loc[self.data_source['reef'] == reef_id]
        return reef
    
    def load_specific_reef(self, reef_id):
        """This method loads a specific reef - order based on dataframe"""
        reef = self.get_specific_reef(reef_id)
        images = []
        for image in reef.full_path:
            geotiff_file_path = image
            geotiff_loader = LoadGeoTIFF(geotiff_file_path)
            data = geotiff_loader.load()
            images.append(data)
        return images

    def load(self):
        data_list = []
        for image in self.data_source.full_path:
                print(image)
                geotiff_file_path = image
                geotiff_loader = LoadGeoTIFF(geotiff_file_path)
                data = geotiff_loader.load()
                data_list.append(data)
        return data_list

class LoadFromZip(DataLoader):

    def __init__(self, data_source):
        super().__init__(data_source)
    
    def load(self):
        # Implement code that will load data from a zip file
        pass


class LoadSeasonalData():
    """Class that loads seasonal composites from a directory based on file name with dates"""
    def __init__(self, directory, substring = "/44_"):
        self.directory = directory
        self.files = list_files_in_dir_recur(self.directory)
        # Temp fix for now to reduce the amount of data being read in - fix later
        self.files =[i for i in self.files if substring in i]

        self.winter_paths = []
        self.summer_paths = []
        self.autumn_paths = []
        self.spring_paths = []
        
        self.dates = self.generate_dates()
        self.seasons = self.gen_seasons()
        self.winter, self.summer, self.autumn, self.spring = self.gen_seasonal_images()

    def generate_seasonal_dates(self, season = "Winter"):
        """This method generates the dates from the data source"""
        dates_dict = {"Winter":self.winter_paths,
                       "Summer":self.summer_paths, 
                       "Autumn":self.autumn_paths,
                         "Spring":self.spring_paths}
        dates = []

        for file in dates_dict[season]:
            dates.append(extract_individual_date(file))
        return dates


    def generate_dates(self):
        """This method generates the dates from the data source"""
        dates = []
        for file in self.files:
            dates.append(extract_individual_date(file))
        return dates


    def gen_seasons(self):
        """This method generates the seasons from the data source"""
        seasons = []
        for date in self.dates:
            seasons.append(get_season(date, southern_hemisphere_meteorological_seasons))
        return seasons

    def gen_seasonal_images(self):
        """This method generates the seasonal images from the data source"""
        winter = []
        summer = []
        autumn = []
        spring = []
        for idx, image, in enumerate(self.files):
            img_season = self.seasons[idx]
            if img_season == "Winter":
                winter.append(image)
                self.winter_paths.append(image)
            elif img_season == "Summer":
                summer.append(image)
                self.summer_paths.append(image)
            elif img_season == "Autumn":
                autumn.append(image)
                self.autumn_paths.append(image)
            elif img_season == "Spring":
                spring.append(image)
                self.spring_paths.append(image)
        return winter, summer, autumn, spring

    def load_images(self, path):
        """ returns geotiff from path"""
        geo_tiff_loader = LoadGeoTIFF(path).load()
        img = np.swapaxes(geo_tiff_loader, 0, 2)
        img = np.swapaxes(img, 0, 1)
        # img = ih.apply_mask(img, np.expand_dims(calculate_water_surface_index(img), axis=2), fill_value = 0)
        # img = ih.apply_mask(img, mask_land(img), fill_value = 0)
        # img = ih.apply_mask(img, np.expand_dims(cloud_index(img) < 0.85, axis=2), fill_value = 0)
        return img


    def load_seasonal_images(self, season):
        """This method loads the seasonal images from the data source"""
        if season == "Winter":
            seasonal_images = self.winter
        elif season == "Summer":
            seasonal_images = self.summer
        elif season == "Autumn":
            seasonal_images = self.autumn
        elif season == "Spring":
            seasonal_images = self.spring
        else:
            raise ValueError("Season not found")
        seasonal_images_data = []
        for image in seasonal_images:
            seasonal_images_data.append(self.load_images(image))
        return seasonal_images_data


if __name__ == "__main__":
    # path_to_hdd = "/run/media/ziad/Expansion/Clipped_Reefs_By_Image/T55LCD_20151204T004052no_transform"
    # file_name = "/6880_T55LCD_20151204T004052no_transform.tiff"
    # full_path = path_to_hdd + file_name

    # dt = LoadGeoTIFF(full_path).load()
    seasonal_loader = LoadSeasonalData("/mnt/sda_mount/Clipped/L1C/")
    dates = seasonal_loader.generate_dates()
    seasonal_images = seasonal_loader.gen_seasonal_images()
    # print(seasonal_images)
    season = 'Winter'
    print(np.array(seasonal_loader.load_seasonal_images(f"{season}")).shape)
    fig,ax = plt.subplots(1,1, figsize=(10,10))
    
    images = seasonal_loader.load_seasonal_images(f"{season}")
    median_images = np.ma.median(images, axis = 0)
    median_images_filled = median_images.filled(np.nan)

    plt.imshow(ih.plot_rgb(median_images_filled) )
    plt.savefig(f"Graphs/{season}.png")