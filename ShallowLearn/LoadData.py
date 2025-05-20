import numpy as np
import rasterio
import csv

from concurrent.futures import ThreadPoolExecutor
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
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from functools import lru_cache
from rasterio.transform import from_bounds

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
        self._subdatasets_cache = None
        self._resolution_subdatasets_cache = None
        self._band_description_cache = None

        if self.is_zip:
            with zipfile.ZipFile(data_source, 'r') as zip_ref:
                self.files = [
                    f for f in zip_ref.namelist()
                    if "MTD_MSIL1C.xml" in f or "MTD_MSIL2A.xml" in f
                ]
            self.zip_path = f"/vsizip/{data_source}"
        elif data_source.endswith(".xml"):
            self.files = [data_source]
        elif data_source.endswith(".SAFE"):
            self.files = [
                i for i in list_files_in_dir_recur(data_source)
                if "MTD_MSIL1C" in i
            ]

        if len(self.files) != 1:
            raise Exception(
                "Multiple or no MTD_MSIL1C files found. Please check your data."
            )

        self.file = os.path.join(self.zip_path, self.files[0]) if self.is_zip else self.files[0]

    def load(self):
        """Cache subdatasets from the primary file."""
        if self._subdatasets_cache is not None:
            return self._subdatasets_cache

        with rasterio.open(self.file) as dataset:
            self._subdatasets_cache = dataset.subdatasets

        with rasterio.open(self._subdatasets_cache[0]) as ds:
            self.tags = ds.tags()
            self.profile = ds.profile
            self.metadata = ds.meta
            self.offsets = ds.offsets
            self.bounds = ds.bounds

        return self._subdatasets_cache

    def get_resolution_subdatasets(self):
        """Cache subdatasets filtered by resolution."""
        if self._resolution_subdatasets_cache is not None:
            return self._resolution_subdatasets_cache

        subdatasets = self.load()
        resolutions = {
            '10m': [s for s in subdatasets if "10m" in s],
            '20m': [s for s in subdatasets if "20m" in s],
            '60m': [s for s in subdatasets if "60m" in s],
            'tci': [s for s in subdatasets if "TCI" in s],
        }
        self._resolution_subdatasets_cache = resolutions
        return resolutions

    def describe_bands(self):
        """Identify bands in each dataset and cache the list."""
        if self._band_description_cache is not None:
            return self._band_description_cache

        resolution_datasets = self.get_resolution_subdatasets()
        description_dict = {}
        for key, items in resolution_datasets.items():
            if key == 'tci' or not items:
                continue
            with rasterio.open(items[0]) as ds:
                desc = ds.descriptions
                description_dict[items[0]] = [d.split(",")[0] for d in desc]

        self._band_description_cache = description_dict
        return description_dict

    def get_selected_bands(self, resolution="10m", selected_bands=None):
        if selected_bands is None:
            selected_bands = ['B02', 'B03', 'B04', 'B08']

        band_description = self.describe_bands()
        resolution_paths = self.get_resolution_subdatasets().get(resolution, [])
        if not resolution_paths:
            raise ValueError(f"No subdatasets found for resolution {resolution}")

        width, height = rs.get_raster_dimensions(resolution_paths[0])
        index_map = {}
        idx_location = len(selected_bands)
        counter = len(selected_bands)

        for band_code, _ in self.band_mapping.items():
            if band_code in selected_bands:
                substring = band_code.replace("0", "") if band_code[1] == "0" else band_code
                for path, desc_list in band_description.items():
                    if substring in desc_list:
                        band_index = desc_list.index(substring) + 1
                        index_map[idx_location - counter] = [path, band_index, band_code]
                        counter -= 1

        return index_map, width, height

    def construct_resampled_array(self, resolution="10m", selected_bands=None):
        if selected_bands is None:
            selected_bands = ['B02', 'B03', 'B04', 'B08']

        band_dict, width, height = self.get_selected_bands(resolution, selected_bands)
        empty_array = np.zeros((width, height, len(selected_bands)))

        for arr_index, band_info in band_dict.items():
            empty_array[:, :, arr_index] = rs.resample_raster(
                band_info[0], band_info[1], width, height
            )
        return empty_array

    def clip_raster_with_shape(self, shapes, resolution='10m', selected_bands=None, use_mask=True):
        """
        Clip the raster with a shape from a shapefile, then resample.
        """
        if selected_bands is None:
            selected_bands = ['B02', 'B03', 'B04', 'B08']

        band_dict, _, _ = self.get_selected_bands(resolution, selected_bands)
        clipped_and_resampled = []
        geometry = [shapes.geometry.iloc[0]] if use_mask else [box(*shapes.geometry.iloc[0].bounds)]
        final_width, final_height = None, None

        # Determine final dimensions
        for _, band_info in band_dict.items():
            if resolution in band_info[0]:
                with rasterio.open(band_info[0]) as src:
                    clipped_img, transform = mask(src, geometry, crop=True)
                    final_height, final_width = clipped_img.shape[1:3]
                break

        if final_width is None or final_height is None:
            raise ValueError("Could not determine final width and height for resampling.")

        # Clip and resample
        for _, band_info in band_dict.items():
            with rasterio.open(band_info[0]) as src:
                clipped_img, transform = mask(src, geometry, crop=True)
                resampled_data = np.empty((final_height, final_width), dtype=clipped_img.dtype)
                new_transform = rasterio.transform.from_bounds(
                    *rasterio.transform.array_bounds(clipped_img.shape[1], clipped_img.shape[2], transform),
                    width=final_width,
                    height=final_height
                )
                reproject(
                    source=clipped_img[band_info[1] - 1],
                    destination=resampled_data,
                    src_transform=transform,
                    src_crs=src.crs,
                    dst_transform=new_transform,
                    dst_crs=shapes.crs,
                    resampling=Resampling.bilinear
                )
                clipped_and_resampled.append(resampled_data)

        return np.array(clipped_and_resampled)


class OptimizedLoadSentinel2L1C:
    """
    Optimized data loader for Sentinel2 L1C. 
    Uses caching to avoid repeated reads and concurrency to speed up clipping/resampling.
    """

    def __init__(self, data_source, band_mapping=band_mapping):
        self.data_source = data_source
        self.band_mapping = band_mapping 
        self.is_zip = data_source.endswith(".zip")
        self._subdatasets_cache = None
        self._resolution_datasets_cache = None
        self._band_description_cache = None

        # If it's a ZIP, find the manifest(s) inside
        mtd_files = []
        if self.is_zip:
            with zipfile.ZipFile(self.data_source, 'r') as zip_ref:
                for fn in zip_ref.namelist():
                    if "MTD_MSIL1C.xml" in fn or "MTD_MSIL2A.xml" in fn:
                        mtd_files.append(fn)
            if len(mtd_files) != 1:
                raise ValueError("Multiple or no metadata files found in the ZIP.")
            self.file_to_open = f"/vsizip/{self.data_source}/{mtd_files[0]}"

        elif self.data_source.endswith(".xml"):
            self.file_to_open = self.data_source
        else:
            raise ValueError("Unsupported data source. Must be .zip or .xml.")

    def load(self):
        """
        Reads subdatasets once and caches. You can expand logic if your subdatasets 
        come from a SAFE directory or other structure.
        """
        if self._subdatasets_cache is not None:
            return self._subdatasets_cache

        with rasterio.open(self.file_to_open) as dataset:
            self._subdatasets_cache = dataset.subdatasets
        return self._subdatasets_cache

    def get_resolution_subdatasets(self):
        """
        Filters subdatasets by resolution. Cached for repeated calls.
        """
        if self._resolution_datasets_cache is not None:
            return self._resolution_datasets_cache

        subs = self.load()
        resolutions = {
            '10m': [s for s in subs if "10m" in s],
            '20m': [s for s in subs if "20m" in s],
            '60m': [s for s in subs if "60m" in s],
            'tci': [s for s in subs if "TCI" in s]
        }
        self._resolution_datasets_cache = resolutions
        return resolutions

    def describe_bands(self):
        """
        Describes band names/identifiers for each resolution subdataset, caching the result.
        """
        if self._band_description_cache is not None:
            return self._band_description_cache

        descriptions = {}
        resolution_dict = self.get_resolution_subdatasets()

        for res_type, paths in resolution_dict.items():
            if not paths or "tci" in res_type:
                continue
            with rasterio.open(paths[0]) as ds:
                # For each subdataset, store the textual descriptions
                descriptions[paths[0]] = [desc.split(',')[0] for desc in ds.descriptions]

        self._band_description_cache = descriptions
        return descriptions

    def get_selected_bands(self, resolution='10m', requested_bands=None):
        """
        Returns a dict of { band_name: subdataset_path } 
        or any structure needed to relate band to subdataset path.
        """
        if requested_bands is None:
            requested_bands = ['B02', 'B03', 'B04', 'B08']

        subdatasets = self.get_resolution_subdatasets().get(resolution, [])
        if not subdatasets:
            raise ValueError(f"No subdatasets for resolution {resolution} found.")

        # Grab the relevant band descriptions
        band_dict = {}
        descriptions = self.describe_bands()
        for sd_path in subdatasets:
            if sd_path not in descriptions:
                continue
            desc_list = descriptions[sd_path]
            # Each descriptive label is like: "B02" or "B03", etc.
            for b in requested_bands:
                # Convert "B08" -> "B8" if needed
                short = b.replace("0", "") if b.startswith("B0") else b
                if short in desc_list:
                    band_dict[b] = sd_path
        return band_dict

    def clip_raster_with_shape(self, shapes, resolution='10m', selected_bands=None, use_mask=True):
        """
        Parallel clip + resample. 
        The returned array will have shape (#bands, height, width) 
        if each clipped result is 3D (count=1, y, x).
        """
        if selected_bands is None:
            selected_bands = ['B02', 'B03', 'B04', 'B08']

        # Map each selected band -> correct subdataset path
        band_dict = self.get_selected_bands(resolution, selected_bands)

        if use_mask:
            # If shapes.geometry is a single shape, you might do shapes.geometry.iloc[0]
            geometry = [shapes.geometry.iloc[0]]
        else:
            # Use bounding box
            geom_box = box(*shapes.geometry.iloc[0].bounds)
            geometry = [geom_box]

        # Strategy: open each band in parallel, clip, and return clipped array
        def process_band(path_b):
            with rasterio.open(path_b) as src:
                clipped_img, transform = mask(src, geometry, crop=True)
                # clipped_img shape: (band_count, height, width)
                return clipped_img

        # ThreadPoolExecutor will release the GIL during I/O
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(process_band, band_dict.values()))

        # results is a list of numpy arrays 
        # each is shape (1, clipped_height, clipped_width)
        return np.concatenate(results, axis=0)

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
        self.files =[i for i in self.files if substring in i and i.endswith(".tiff")]

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
        if path.endswith(".tiff"):
            path = path
        else:
            ValueError("Path does not end with .tiff")

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