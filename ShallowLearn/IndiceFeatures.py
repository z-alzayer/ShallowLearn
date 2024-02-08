import inspect
import importlib
import numpy as np
import ShallowLearn.Indices as band_indices

import matplotlib.pyplot as plt


def insert_empty_channel(image, index):
    """
    Insert an empty channel into a multi-channel image at the specified index.

    Parameters:
    image (numpy.ndarray): A multi-channel image (3D array).
    index (int): The index at which to insert the empty channel.

    Returns:
    numpy.ndarray: The modified image with an additional empty channel.
    """
    if index < 0 or index > image.shape[2]:
        raise ValueError("Index is out of bounds for the image channels.")

    # Create an empty channel with zeros
    height, width, _ = image.shape
    empty_channel = np.zeros((height, width))

    # Insert the empty channel
    image_with_channel = np.insert(image, index, empty_channel, axis=2)

    return image_with_channel


class GenerateIndicesPerImage():
    """
    Class for generating indices per image.
    
    """
    def __init__(self, img ):
        
        """Takes in loaded tiff using load_img method from ShallowLearn.ImageHelper.py
        and generates indices for each band in the image. as defined in the init method."""


        self.img = img
        print(self.img.shape)
        if self.img.shape[-1] <= 12:
            #work around for s2a data
            self.img = insert_empty_channel(self.img, 9)

        self.functions = get_feature_order()
        self.indices, self.indice_order = self.generate_indices()

    def generate_indices(self):
        """Generates the indices for each band in the image and returns a numpy array of the indices."""
        arr = []
        indice_order = []

        for function in self.functions:
            indice_order.append(function[0])
            # print(function[0])
            arr.append(function[1](self.img))

        arr = np.array(arr)
        arr = np.swapaxes(arr, 0, 2)
        arr = np.swapaxes(arr, 0, 1)    
        return arr, indice_order

def get_feature_order():
    # TODO: build a dictionary that does this automatically

    module = importlib.import_module('ShallowLearn.Indices')
    members = inspect.getmembers(module)
    functions = [member for member in members if inspect.isfunction(member[1])]

    functions = [function for function in functions if 
        function[0] != 'validate_band_shape' and
        function[0] != 'get_band_numbers' and 
        function[0] != 'mask_land' and
        function[0] != "remove_zeros_decorator" and
        len(function[0]) < 5 or 
        function[0] == "calculate_pseudo_subsurface_depth"
        or function[0] == "calculate_water_surface_index"
        or function[0] == 'cl_oci'
       ]
    return functions

    

from ShallowLearn.ImageHelper import load_img

if __name__ == "__main__":
    path = "/media/ziad/Expansion/Cleaned_Data_Directory/"
    img = load_img(path + "6880_T55LCD_20220401T003711no_transform.tiff")
    GenerateIndicesPerImage(img)