import numpy as np
import rasterio
import csv
import pandas as pd
class DataLoader:
    def __init__(self, data_source):
        self.data_source = data_source

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

    def load(self):
        # Implement the method to load a GeoTIFF file using Rasterio
        with rasterio.open(self.data_source) as src:
            data = src.read()
        return data

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


if __name__ == "__main__":
    path_to_hdd = "/run/media/ziad/Expansion/Clipped_Reefs_By_Image/T55LCD_20151204T004052no_transform"
    file_name = "/6880_T55LCD_20151204T004052no_transform.tiff"
    full_path = path_to_hdd + file_name

    dt = LoadGeoTIFF(full_path).load()
