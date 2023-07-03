import inspect
import importlib
import numpy as np
import ShallowLearn.Indices as band_indices

import matplotlib.pyplot as plt


class GenerateIndicesPerImage():
    """
    Class for generating indices per image.
    
    """
    def __init__(self, img):
        self.img = img
        module = importlib.import_module('ShallowLearn.Indices')
        members = inspect.getmembers(module)
        functions = [member for member in members if inspect.isfunction(member[1])]
        print(functions)
        functions = [function for function in functions if 
                     function[0] != 'validate_band_shape' and
                        function[0] != 'get_band_numbers' and 
                        function[0] != 'mask_land']
        self.functions = functions
        print(functions)
        self.indices, self.indice_order = self.generate_indices()
        print(self.indice_order)


    def generate_indices(self):
        arr = []
        indice_order = []
        for function in self.functions:
            indice_order.append(function[0])
            arr.append(function[1](self.img))
        arr = np.swapaxes(np.array(arr), 0, 2)
        arr = np.swapaxes(arr, 0, 1)    
        return arr, indice_order

from ShallowLearn.ImageHelper import load_img

if __name__ == "__main__":
    path = "/media/ziad/Expansion/Cleaned_Data_Directory/"
    img = load_img(path + "6880_T55LCD_20220401T003711no_transform.tiff")
    GenerateIndicesPerImage(img)