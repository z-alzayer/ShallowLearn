import ShallowLearn.FileProcessing as fp
import ShallowLearn.ImageHelper as ih
import rasterio as rio 
from rasterio.enums import Resampling
import numpy as np
import os 
import matplotlib.pyplot as plt 

CLOUD = 2
CLOUD_SHADOW = 3
SNOW = 4
NULL = 0

def resample_mask(mask_path, upscale_factor = 2):
    """
    Resamples a mask to a higher resolution using nearest neighbor resampling.
    Args:
        mask_path (str): The path to the mask file.
        upscale_factor (int): The factor to upscale the mask by.
    Returns:
        np.ndarray: The resampled mask.
        transform: The transform of the resampled mask.
    
    """
    upscale_factor = upscale_factor
    with rio.open(mask_path) as dataset:

        # resample data to target shape
        data = dataset.read(
            out_shape=(
                dataset.count,
                int(dataset.height * upscale_factor),
                int(dataset.width * upscale_factor)
            ),
            resampling=Resampling.nearest
        )

        # scale image transform
        transform = dataset.transform * dataset.transform.scale(
            (dataset.width / data.shape[-1]),
            (dataset.height / data.shape[-2])
        )
    return data, transform 

def reshape_mask(img):
    """
    swaps the channel with the height and width of the image so that channel
    is on the last axes of the image.
    """
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 0, 1)
    return img

def preproc_mask(mask):
    """
    fmask specific preprocessing to remove clouds and shadows from 
    the mask
    Args:
        mask (np.ndarray): The mask to be preprocessed.
    Returns:
        np.ndarray: The preprocessed mask.
    """
    mask = (mask != CLOUD) & (mask != CLOUD_SHADOW) & (mask != NULL) & (mask != SNOW)
    return mask



def apply_mask(image, mask):
    """
    Applies a mask to an image.
    Args:
        image (np.ndarray): The image to be masked.
        mask (np.ndarray): The mask to be applied.
    Returns:
        np.ndarray: The masked image.
    """
    masked_image = ih.apply_mask(image, mask)
    return masked_image

def apply_fmask_from_path(image_path, mask_path, upscale_factor = 2):
    """
    Applies a mask to an image using the path to the image and the mask.
    Args:
        image_path (str): The path to the image.
        mask_path (str): The path to the mask.
        upscale_factor (int): The factor to upscale the mask by.
    Returns:
        np.ndarray: The masked image.
        transform: The transform of the resampled mask.
        meta: The metadata of the image.
        bounds: The bounds of the image.

    """
    mask, transform = resample_mask(mask_path, upscale_factor)
    reshaped_mask = reshape_mask(mask)
    image, meta, bounds = ih.load_img(image_path, return_meta = True)
    masked_image = apply_mask(image, preproc_mask(reshaped_mask))
    return masked_image, transform, meta, bounds


def save_masked_file(image_path, mask_path, output_path, upscale_factor = 2):
    """
    Applies a mask to an image and saves the masked image to a file.
    Args:
        image_path (str): The path to the image.
        mask_path (str): The path to the mask.
        output_path (str): The path to save the masked image.
        upscale_factor (int): The factor to upscale the mask by.
    """
    masked_image, transform, metadata, bounds = apply_fmask_from_path(image_path, mask_path, upscale_factor)

    masked_image = np.swapaxes(masked_image, 0, 1)
    masked_image = np.swapaxes(masked_image, 0, 2)


    metadata.update({'transform': transform, 'height': masked_image.shape[1], 'width': masked_image.shape[2]})
    
    with rio.open(f"{output_path}_fmasked.tiff", 'w', **metadata) as dst:
        dst.write(masked_image)
        print("Finished writing file")

def apply_mask_l1c(image_path, mask_path):
    """
    Applies a mask to an image.
    Args:
        image (np.ndarray): The image to be masked.
        mask (np.ndarray): The mask to be applied.
    Returns:
        np.ndarray: The masked image.
    """
    image = ih.load_img(image_path)
    mask = ih.load_img(mask_path)

    masked_image = ih.apply_mask(image, mask)
    return masked_image




if __name__ == "__main__":
    # initial tests on subset of image
    # path_image = "/mnt/sda_mount/L1C/S2A_MSIL1C_20160323T003752_N0201_R059_T55LCD_20160323T003830.SAFE.tiff"
    # path_fmask = '/media/zba21/Expansion/Cloud_Masks/fmask_S2A_MSIL1C_20160323T003752_N0201_R059_T55LCD_20160323T003830.SAFE'

    # mask, transform = resample_mask(path_fmask, upscale_factor = 2)
    # reshaped_mask = reshape_mask(mask)
    # processed = preproc_mask(reshaped_mask)
    # image = ih.load_img(path_image)
    # print(processed.shape)
    # masked_image = apply_mask(image[3000:4000, 3000:4000,:], reshaped_mask[3000:4000, 3000:4000])
    # print(masked_image.shape)
    # masked_image, transform = apply_fmask_from_path(path_image, path_fmask, upscale_factor = 2)
    # plt.imshow(ih.plot_rgb(masked_image))
    # plt.show()

    # Code to clip from fmask files
    # counter = 0

    # file_name_dictionary = {}
    # l1c_scenes = fp.list_files_in_dir_recur("/mnt/sda_mount/L1C/")
    # path = "/media/zba21/Expansion/Cloud_Masks/"
    # files = [i for i in fp.list_files_in_dir_recur(path) if i.endswith(".SAFE")]

    # for mask_path in files:
    #     for i in l1c_scenes:
    #         mask_date = mask_path.split("/")[-1].split("_")[-1].split(".")[0]
    #         if fp.check_in_string(i,extension='tiff', other_string=mask_date):
    #             print(i)
    #             file_name_dictionary[i] = mask_path
    #             counter += 1
    # print(counter)

    # for key, value in file_name_dictionary.items():
    #     outpath = key.split("/")[-1]
        # save_masked_file(key, value, f"/mnt/sda_mount/fmasked/{outpath}", upscale_factor = 2)

    # save_masked_file(path_image, path_fmask, "/mnt/sda_mount/fmasked/", upscale_factor = 2)
    


    # Code to clip from sentinel mask files
    # counter = 0
    # l1c_file_name_dict = {}
    # l1c_scenes = fp.list_files_in_dir_recur("/mnt/sda_mount/L1C/")
    # path = "/mnt/sda_mount/L1C_Cloud_Masks/"
    # files = [i for i in fp.list_files_in_dir_recur(path) if i.endswith(".tif")]

    # for mask_path in files:
    #     for i in l1c_scenes:
    #         mask_date = mask_path.split("/")[-1].split("_")[2].split(".")[0]
    #         if fp.check_in_string(i,extension='tiff', other_string=mask_date):
    #             # print(i)
    #             l1c_file_name_dict[i] = mask_path
    #             counter += 1
    #             # mask_applied = apply_mask_l1c(i, mask_path)
    #             # ih.plot_rgb(mask_applied)
    #             # plt.show()
    #             print(l1c_file_name_dict[i])
    #             break
    #     break

    # print(counter)