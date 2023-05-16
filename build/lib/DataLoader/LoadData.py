import numpy as np
import rasterio
import csv

class LoadData:
    def __init__(self, data_source):
        self.data_source = data_source

    def load(self):
        raise NotImplementedError("This method should be implemented by the subclass")

class LoadRemoteSensingData(LoadData):
    """Loads a numpy array of remotely sensed data"""
    def __init__(self, data_source):
        super().__init__(data_source)

    def load(self):
        # Implement the method to load remote sensing data using NumPy
        data = np.load(self.data_source)
        return data

class LoadGeoTIFF(LoadData):
    """Loads a geotiff directly from a raster accepted by rasterio"""
    def __init__(self, data_source):
        super().__init__(data_source)

    def load(self):
        # Implement the method to load a GeoTIFF file using Rasterio
        with rasterio.open(self.data_source) as src:
            data = src.read()
        return data

class LoadDataFromCSV(LoadData):
    """This class loads data from a CSV file. The CSV file should contain the file paths of the data to be loaded.
        It assumes that the file paths are in the first column of the CSV file."""
    def __init__(self, data_source):
        super().__init__(data_source)

    def load(self):
        data_list = []
        with open(self.data_source, newline='') as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                geotiff_file_path = row[0]  # Assuming the first column contains the GeoTIFF file path
                geotiff_loader = LoadGeoTIFF(geotiff_file_path)
                data = geotiff_loader.load()
                data_list.append(data)
        return data_list