from rasterio import crs
from rasterio.enums import Resampling
from rasterio import open as rio_open
import numpy as np
import logging
import os
import tempfile
import shutil


from ShallowLearn.LoadData import LoadGeoTIFF
import ShallowLearn.Transform as tf
from ShallowLearn.FileProcessing import (unzip_files, get_file_names_from_zip, delete_files_from_dir, 
                                         list_files_in_dir, filter_files_by_extension, check_values_in_filenames,
                                        order_by_band, list_files_in_dir_recur, order_band_names_noreg)
from ShallowLearn.band_mapping import band_mapping

files_to_keep = []


class ImageCompiler():
    """
    Class for compiling a set of images into one multi-band image.

    Attributes
    ----------
    image_paths : list
        Paths of the images to compile.
    highest_resolution_band : str
        The path of the band with the highest resolution.
    output_path : str
        Path where the output should be saved.
    """

    def __init__(self, image_paths, highest_resolution_band, output_path):
        """Create a new ImageCompiler instance."""
        self.image_paths = image_paths
        self.highest_resolution_band = highest_resolution_band
        self.output_path = output_path

        self.get_common_metadata()

    def get_common_metadata(self):
        """Fetch common metadata from the highest resolution band."""
        self.highest_resolution_band = rio_open(self.highest_resolution_band)
        self.crs = self.highest_resolution_band.crs
        self.height = self.highest_resolution_band.height
        self.width = self.highest_resolution_band.width
        self.transform = self.highest_resolution_band.transform
        self.count = self.highest_resolution_band.count
        self.bands = len(self.image_paths)
        self.dtype = self.highest_resolution_band.dtypes[0]
    
    def create_stack(self, save_name):
        """
        Initialize a stack for compiling images.

        Parameters
        ----------
        save_name : str
            The name to use when saving the compiled image.
        """
        self.stacked_image = rio_open(save_name, 'w', driver='Gtiff',
        width=self.width, height=self.height,
        count=self.bands,
        crs=self.crs,
        transform=self.transform,
        dtype=self.dtype
        )

    def compile(self):
        """Compile the images into a single stacked image."""
        for i in range(self.bands):
            with rio_open(self.image_paths[i]) as src:
                # Resample if dimensions are not the same as the highest resolution
                if src.width <  self.width or src.height < self.height:
                    resampled_image = src.read(out_shape=(self.height, self.width), resampling=Resampling.cubic)[0, :, :]
                    self.stacked_image.write(resampled_image, i + 1)
                else:
                    self.stacked_image.write(src.read(1), i+1)
        self.stacked_image.close()


class GeotiffGenerator():
    """
    Class for generating GeoTIFF files from Sentinel-2 zip or extracted zip files.

    Attributes
    ----------
    zip_path : str
        Path of the zip file to process.
    output_path : str
        Path where the output GeoTIFF should be saved.
    band_order : dict
        Order to use when processing the bands.
    output_name : str
        Name to use when saving the output GeoTIFF.
    """

    def __init__(self, zip_path, output_path, output_name, band_order = band_mapping):
        """Create a new GeotiffGenerator instance."""
        self.zip_path = zip_path
        self.output_path = output_path
        self.band_order = band_order
        self.output_name = output_name
        # self.process_zip()

    def process_extracted_zip(self, high_res_band = "B02"):
        """
        Process the extracted zip file and generate a GeoTIFF.

        Parameters
        ----------
        high_res_band : str
            The band with the highest resolution to use when processing.
            Defaults to "B02".
        """
        # Filter out the image files from the zip
        image_files = list_files_in_dir_recur(self.zip_path)
        image_files = check_values_in_filenames(image_files, self.band_order)
        
        # This is super sentinel specific - needs to be changed for landsat or atleast made more general
        self.ordered_image_files = [file for file in image_files if "IMG_DATA" in file]


        ##### fix the stuff here
        # print(order_by_band_wo_regex(self.ordered_image_files))
        # Append the outpath to the extracted image files
        self.ordered_image_files = order_band_names_noreg(self.ordered_image_files)
        # for i in self.ordered_image_files:
        #     print(i)
        # print(self.ordered_image_files)
        high_res_index = band_mapping[high_res_band]['index']
        self.stack = ImageCompiler(self.ordered_image_files, self.ordered_image_files[2], self.output_path + self.output_name)
        self.stack.create_stack(self.output_path + self.output_name)
        self.stack.compile()
    
    def process_sen2cor_local(self, high_res_band = "B02"):
        """
        Process the extracted zip file and generate a GeoTIFF.

        Parameters
        ----------
        high_res_band : str
            The band with the highest resolution to use when processing.
            Defaults to "B02".
        """
        # Filter out the image files from the zip
        image_files = list_files_in_dir_recur(self.zip_path)
        # print(image_files)
        image_files = check_values_in_filenames(image_files, self.band_order)
        # print(image_files)
        # This is super sentinel specific - needs to be changed for landsat or atleast made more general
        self.ordered_image_files = [file for file in image_files if "IMG_DATA" in file]

        # print(self.ordered_image_files)
        ##### fix the stuff here
        # print(order_by_band_wo_regex(self.ordered_image_files))
        # Append the outpath to the extracted image files
        self.ordered_image_files = order_by_band(self.ordered_image_files)
        # for i in self.ordered_image_files:
        #     print(i)
        # print(self.ordered_image_files)
        high_res_index = band_mapping[high_res_band]['index']
        self.stack = ImageCompiler(self.ordered_image_files, self.ordered_image_files[2], self.output_path + self.output_name)
        self.stack.create_stack(self.output_path + self.output_name)
        self.stack.compile()



    def process_zip(self, high_res_band = "B02"):
        """
        Process the zip file and generate a GeoTIFF.
        This uses L2A Imagery that is already processed by ESA

        Parameters
        ----------
        high_res_band : str
            The band with the highest resolution to use when processing.
            Defaults to "B02".
        """
        # Filter out the image files from the zip
        image_files = filter_files_by_extension(get_file_names_from_zip(self.zip_path), ".jp2")
        image_files = check_values_in_filenames(image_files, self.band_order)
        # This is super sentinel specific - needs to be changed for landsat or atleast made more general
        self.ordered_image_files = order_by_band([file for file in image_files if "/IMG_DATA" in file])
        
        unzip_files(self.zip_path, self.ordered_image_files,  self.output_path)
        # Append the outpath to the extracted image files
        self.ordered_image_files = [self.output_path + i for i in self.ordered_image_files]
        
        high_res_index = band_mapping[high_res_band]['index']
        self.stack = ImageCompiler(self.ordered_image_files, self.ordered_image_files[2], self.output_name)
        self.stack.create_stack(self.output_name)
        self.stack.compile()
        # print(self.output_path)
        # print(self.output_path + self.output_name)
        if "/" in self.output_name:
            self.output_name = self.output_name.split("/")[-1]
            files_to_keep.append(self.output_name)
        print(files_to_keep)
        delete_files_from_dir(self.output_path, files_to_keep)

if __name__ == "__main__":
    import os
    import pandas as pd

    print(os.getcwd())
    
    path = "/media/ziad/Expansion/Full_Imagery/S2A_Conversion"
    suitable_imagery = pd.read_csv("Data/Cloud_Mask_40_threshold.csv")
    
    img_list = list(suitable_imagery.to_dict()['0'].values())
    acquisition_date = [name.split("_")[2] for name in img_list]
    # for name in img_list:
    #     print(name.split("_")[2])
  
    imagery = os.listdir(path)
    suitable_files = check_values_in_filenames(imagery, acquisition_date)
    print(len(suitable_files))
    # for i in imagery:
    #     if i.endswith(".SAFE"):
    #         date_number = i.split('_')[2][:8]  # Extract the date number, e.g., 20220307
            
    #         # Check if date_number exists in any of the filenames in img_list
    #         if any(date_number in img_name for img_name in img_list):
    #             print(i)
    #             # img_gen = GeotiffGenerator(os.path.join(path, i), 
    #             #                            "/media/ziad/Expansion/Full_Imagery/CompiledImagery/",
    #             #                            f"{i}.tiff")
    #             # img_gen.process_sen2cor_local()