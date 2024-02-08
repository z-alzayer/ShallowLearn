import numpy as np
import rasterio
import csv
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

from ShallowLearn.FileProcessing import list_files_in_dir_recur
from ShallowLearn.DateHelper import extract_dates, get_season, southern_hemisphere_meteorological_seasons, extract_individual_date
import ShallowLearn.ImageHelper as ih
from ShallowLearn.PreprocDecorators import remove_zeros_decorator
from ShallowLearn.Indices import calculate_water_surface_index, mask_land, cloud_index

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
        img = ih.apply_mask(img, mask_land(img), fill_value = 0)
        img = ih.apply_mask(img, np.expand_dims(cloud_index(img) < 0.85, axis=2), fill_value = 0)
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