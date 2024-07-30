from ukis_csmask.mask import CSmask
import rasterio as rio 
import numpy as np

import ShallowLearn.ImageHelper as ih
import ShallowLearn.FileProcessing as fp


def generate_csmask(image):
    csmask = CSmask(
    img=image[:,:,[2,3,4,8,11,12]].astype(np.float32),
    band_order=["blue", "green", "red", "nir", "swir16", "swir22"],
    nodata_value=0,
    )

    # access cloud and cloud shadow mask
    csmask_csm = csmask.csm

    # access valid mask
    csmask_valid = csmask.valid
    return csmask_csm, csmask_valid

def apply_csmask_from_path(image_path):
    """
    Applies a mask to an image using the path to the image and the mask.
    Args:
        image_path (str): The path to the image.
        mask_path (str): The path to the mask.
    Returns:
        np.ndarray: The masked image.
        transform: The transform of the resampled mask.
        meta: The metadata of the image.
        bounds: The bounds of the image.

    """
    image, meta, bounds = ih.load_img(image_path, return_meta = True)
    _ , valid_pixels = generate_csmask(image)
    masked_image = ih.apply_mask(image, valid_pixels)
    return masked_image, meta, bounds





def save_masked_file(image_path, output_path):
    """
    Applies a mask to an image and saves the masked image to a file.
    Args:
        image_path (str): The path to the image.
        mask_path (str): The path to the mask.
        output_path (str): The path to save the masked image.
        upscale_factor (int): The factor to upscale the mask by.
    """
    masked_image, metadata, bounds = apply_csmask_from_path(image_path)

    masked_image = np.swapaxes(masked_image, 0, 1)
    masked_image = np.swapaxes(masked_image, 0, 2)
    
    with rio.open(f"{output_path}", 'w', **metadata) as dst:
        dst.write(masked_image)
        print("Finished writing file")


if __name__ == "__main__":
    files = fp.list_files_in_dir_recur("/mnt/sda_mount/L1C/")
    ukis_path = "/mnt/sda_mount/ukis_masks/"
    for file in files:
        print(f"{ukis_path}{file.split('/')[-1].split(".tiff")[0]}_csmask.tiff")
        save_masked_file(file, f"{ukis_path}{file.split('/')[-1].split(".tiff")[0]}_csmask.tiff")