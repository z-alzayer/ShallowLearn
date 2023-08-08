# Class to generate indices for a time series of images and store the image masks and indices in a numpy array.

import numpy as np
import os

import ShallowLearn.Indices as band_indices
from ShallowLearn.ImageHelper import load_img
from ShallowLearn.IndiceFeatures import GenerateIndicesPerImage
from ShallowLearn.FileProcessing import check_in_string

class GenerateIndicesPerTimeSeries():
    """Helper class that generates indices, masks and images for a time series of sentinel-2 tiff images."""
    
    def __init__(self, path, tag = None, save = True):
        self.path = path
        self.tag = tag
        self.imgs = self.get_imgs(self.tag)
        self.indices = self.generate_indices()
        self.masks = self.generate_masks()
        
        if save:
            self.save_data()
    
    def get_imgs(self, tag):
        """Returns a numpy array of tiff images in the directory."""

        imgs = []
        if tag == None:
            for file in os.listdir(self.path):
                if file.endswith(".tiff"):
                    imgs.append(load_img(self.path + file))
            return np.array(imgs)
        
        for file in os.listdir(self.path):
            if check_in_string(file,extension="tiff", other_string=tag):
                    print(file)
                    imgs.append(load_img(self.path + file))
        return np.array(imgs)
    
    def generate_masks(self):
        """Returns a numpy array of masks for each image in the time series."""

        masks = []
        for img in self.imgs:
            masks.append(band_indices.mask_land(img))
        return np.array(masks)


    def generate_indices(self):
        """Returns a numpy array of indices for each image in the time series."""

        indices = []
        for img in self.imgs:
            indices.append(GenerateIndicesPerImage(img).indices)
        return np.array(indices)

    def save_data(self, path = None):
        """Saves the indices, masks and images to the path specified."""
        if path != None:
            self.path = path
        np.save(self.path + "imgs.npy", self.imgs)
        np.save(self.path + "indices.npy", self.indices)
        np.save(self.path + "masks.npy", self.masks)


if __name__ == "__main__":
    path = "/media/ziad/Expansion/Cleaned_Data_Directory/"
    
    GenerateIndicesPerTimeSeries(path, "6880")