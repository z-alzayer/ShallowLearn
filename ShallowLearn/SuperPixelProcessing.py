import ShallowLearn.SuperPixelExtraction as spe
import ShallowLearn.ImageHelper as ih
import ShallowLearn.FileProcessing as fp
import ShallowLearn.CloudDetector as cloud_detector
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from skimage.color import rgb2lab
import ShallowLearn.Transform as trf
import joblib
import os
from sklearn.mixture import BayesianGaussianMixture
import time
from scipy import stats
from skimage import exposure
import re

def extract_date_from_filename(filename):
    """ Extract the date from the filename string """
    match = re.search(r'\d{8}', filename)
    return match.group(0) if match else 'Unknown Date'


def calculate_image_statistics(image, statistic_type='mean'):
    """
    Calculates and returns the mean, mode, or median of an image per channel.

    Parameters:
    - image (np.ndarray): A NumPy array representing the image (width, height, channels).
    - statistic_type (str): The type of statistic to calculate ('mean', 'mode', 'median').

    Returns:
    - np.ndarray: An array of the calculated statistics for each channel.
    """
    statistics = np.zeros(image.shape[2])

    for channel in range(image.shape[2]):
        if statistic_type == 'mean':
            statistics[channel] = image[:, :, channel].mean()
        elif statistic_type == 'median':
            statistics[channel] = np.median(image[:, :, channel])
        elif statistic_type == 'mode':
            mode_result = stats.mode(image[:, :, channel], axis=None)
            statistics[channel] = mode_result.mode[0]
        else:
            raise ValueError("Invalid statistic type specified. Choose 'mean', 'mode', or 'median'.")

    return statistics

def calculate_nonzero_percentile(data, percentile):
    """
    Calculate a specific percentile from a 1D array after removing zeros.

    Args:
    - data (numpy array): The input 1D array from which to remove zeros.
    - percentile (float): The percentile to calculate (value between 0 and 100).

    Returns:
    - float: The computed percentile value from the non-zero elements of the array.
    """
    # Remove zeros
    non_zero_data = data[data != 0]
    
    # Check if there are any non-zero elements left
    if non_zero_data.size == 0:
        return 0
    
    # Calculate percentile
    percentile_value = np.percentile(non_zero_data, percentile)
    
    return percentile_value



def gen_dii(image, sup, deep_label=4, get_deep = False):
    # Print unique superpixel labels to verify input
    # print(np.unique(sup))
    # Determine the masks for deep and shallow regions based on the deep_label
    mask_deep = sup == deep_label
    mask_shallow = ~mask_deep  # All other labels are considered shallow

    # Create an empty array to store DII results for each band
    dii_results = np.empty_like(image, dtype=np.float32)

    # Process each band
    deep_areas = []
    shallow_areas = []
    for band in range(image.shape[2]):  # Assuming image has shape [height, width, bands]
        # Extract band data
        band_data = image[:, :, band]
        
        # Calculate average reflectance in deep and shallow areas for the current band
        try:
            print(band_data[mask_deep].shape)
            R_deep_avg = calculate_nonzero_percentile(band_data[mask_deep], 0.9)
            deep_areas.append(R_deep_avg)
        except Exception as e:
            print(f"Unable to get a percentile due to: {str(e)}")
            return None
        
        R_shallow_avg = np.mean(band_data[mask_shallow])
        shallow_areas.append(R_shallow_avg)
        # Compute DII for each pixel in the current band
        R_diff = band_data - R_deep_avg
        dii_band = np.where(R_diff > 0, np.log(np.maximum(R_diff, 1e-10)), 0)

        # Store the DII result for the current band
        dii_results[:, :, band] = dii_band
    if get_deep:
        return deep_areas, shallow_areas

    return dii_results
# Function to load models
def load_models():
    pca = joblib.load('../../Models/SuperPixelTS.pkl')
    bgm = joblib.load("../../Models/BayesianGMSuperPixel.pkl")
    return pca, bgm

# Function to process cloud in the image
def process_cloud(image_path):
    print("Processing cloud")
    img = cloud_detector.cloud_regressor(image_path, threshold=300)
    return img


# Function to plot results
def plot_results(dii_results, img_rescale, img, image_path):
    fig, ax = plt.subplots(1, 3, figsize=(10, 10))
    ax[0].imshow(ih.plot_rgb(dii_results))
    ax[1].imshow(ih.plot_rgb(img_rescale))
    ax[2].imshow(ih.plot_rgb(img))
    ax[0].set_title("DII Results")
    ax[1].set_title("Exposure Adjusted")
    ax[2].set_title("Cloud Removed Image")
    fname_substring = extract_date_from_filename(image_path)
    print(f"Finished processing: {fname_substring}")
    plt.show()

def create_and_pad_superpixels_v2(img, shape = (32, 32, 13)):
    lab_image = rgb2lab(img[:,:,[3,2,1]] / 10_000) # revert 10_000 later
    super_px = spe.slic_segmentation(lab_image, n_segments = int(img.shape[0]/2), faster_slic=False)
    padded_segments = spe.pad_slice_segments(img, super_px, shape = shape)
    return padded_segments, super_px

def create_and_pad_super_pixels(img):
    start_time = time.time()
    lab_image = ih.plot_lab(img).astype(np.uint8)
    super_pixel = spe.slic_segmentation(lab_image, n_segments=int(img.shape[0] / 2))
    padded_segment = spe.pad_slice_segments(img, super_pixel)
    end_time = time.time()
    print(f"Processing time: {end_time - start_time} seconds")
    return padded_segment, super_pixel

def apply_pca_bgm(padded_segment, pca, bgm):
    padded_array = np.array(padded_segment)
    transformed_pca = pca.transform(padded_array.reshape(padded_array.shape[0], -1))
    labels = bgm.predict(transformed_pca)
    return labels

def relabel_super_pixels(super_pixel, labels):
    # Create an output array for relabeled super pixels
    relabeled_super_pixels = np.zeros_like(super_pixel)
    # Relabel each super pixel according to the new labels
    for unique_label in np.unique(super_pixel):
        try:
            relabeled_super_pixels[super_pixel == unique_label] = labels[unique_label]
        except:
            print(f"Index error at super pixel label{unique_label}")

    return relabeled_super_pixels

def process_single_image_from_array(img, get_deep = False):
    pca,bgm = load_models()

    padded_segment, super_pixel = create_and_pad_superpixels_v2(img)
        
    labels = apply_pca_bgm(padded_segment, pca, bgm)
    relabeled_super_pixels = relabel_super_pixels(super_pixel, labels)
        # Additional processing...
    try:
        dii = gen_dii(img, relabeled_super_pixels, deep_label=4, get_deep=get_deep)
        if get_deep == True:
            return dii, relabeled_super_pixels
        dii_results = ih.apply_mask(dii, spe.ostu_filter(dii))

        return dii_results
    except Exception as e:
        print(f"Processing failed: {str(e)}")

    print("Single image processing complete")

def process_single_image(image_path, plot=False, get_deep = False):
    print(os.path.curdir)
    pca, bgm = load_models()
    
    if image_path.endswith(".tiff"):
        img = process_cloud(image_path)

        padded_segment, super_pixel = create_and_pad_super_pixels(img)
        
        labels = apply_pca_bgm(padded_segment, pca, bgm)
        relabeled_super_pixels = relabel_super_pixels(super_pixel, labels)
        
        # Additional processing...
        try:
            dii = gen_dii(img, relabeled_super_pixels, deep_label=4, get_deep=get_deep)
            if get_deep == True:
                return dii, relabeled_super_pixels
            dii_results = ih.apply_mask(dii, spe.ostu_filter(dii))
            img_rescale = exposure.equalize_hist(dii_results)
            if plot:
                plot_results(dii_results, img_rescale, img, image_path)
            return dii_results
        except Exception as e:
            print(f"Processing failed: {str(e)}")

        print("Single image processing complete")


if __name__ == "__main__":
    print(os.path.curdir)
    pca = joblib.load('Models/SuperPixelTS.pkl')
    bgm = joblib.load("Models/BayesianGMSuperPixel.pkl")

    path = "/mnt/sda_mount/Clipped/L1C/"
    images = fp.list_files_in_dir_recur(path)
    images = [i for i in images if i.endswith(".tiff")]

    IMG_STR = "24_"
    # IMG_STR = "34_"
    images = [i for i in images if IMG_STR in i]
    cloud_less = []
    print(len(images))
    # img = ih.load_img(images[0])
    # plt.imshow(ih.plot_rgb(img))
    # plt.show()
    print("Processing clouds")
    counter = 0
    for image in images:
        img = cloud_detector.cloud_regressor(image)
        cloud_less.append(img)
        if counter == 10:
            break
        print(img.shape)
        # break
        # break
    img_arr = np.array(cloud_less)
    np.save("/home/zba21/Documents/ShallowLearn/Data/24_Reef/cloudless_arr.npy", img_arr)
    super_pixels = []
    padded_segments = []

    start_time = time.time()  # Start timing before the loop
    print("Creating super pixels")
    for image in img_arr:
        # Assuming ih.plot_lab(image) processes and returns image data
        lab_image = ih.plot_lab(image).astype(np.uint8)
        super_pixels.append(spe.slic_segmentation(lab_image, n_segments = int(img.shape[0] / 2)))
        padded_segments.extend(spe.pad_slice_segments(image, super_pixels[-1]))

    end_time = time.time()  # End timing after the loop
    elapsed_time = end_time - start_time  # Calculate the elapsed time

    print(f"Processing time: {elapsed_time} seconds")

    # start_time = time.time()  # Start timing before the loop

    # for image in img_arr:
    #     # Assuming ih.plot_lab(image) processes and returns image data
    #     lab_image = ih.plot_lab(image).astype(np.uint8)
    #     super_pixels.append(spe.slic_segmentation(lab_image, faster_slic=False))
    #     padded_segments.extend(spe.pad_slice_segments(image, super_pixels[-1]))

    # end_time = time.time()  # End timing after the loop
    # elapsed_time = end_time - start_time  # Calculate the elapsed time

    # print(f"Processing time: {elapsed_time} seconds")

    # for sup, image in zip(super_pixels, img_arr):
        

    padded_array = np.array(padded_segments)

    transformed_pca = pca.transform(padded_array.reshape(padded_array.shape[0], -1))
    # bgm = BayesianGaussianMixture(n_components=5).fit(transformed_pca)
    labels = bgm.predict(transformed_pca)
    # joblib.dump(bgm, "Models/BayesianGMSuperPixel.pkl")
    np.save("/home/zba21/Documents/ShallowLearn/Data/24_Reef/SuperPixelArr.npy", np.array(super_pixels))
    dii_arr = []
    final_paths = []
    for i in range(70):
        try:
            dii = gen_dii(img_arr[i], super_pixels[i], deep_label = 5)
            dii_results = ih.apply_mask(dii, spe.ostu_filter(dii))
            dii_arr.append(dii_results)
            # if i < 7:
            #     continue
            # dii_results = trf.LCE_multi(dii_results)
            fig, ax = plt.subplots(1,3, figsize = (10, 10))
            # p2, p98 = np.percentile(dii_results, (2, 98))
            img_rescale = exposure.equalize_hist(dii_results)
            ax[0].imshow(ih.plot_rgb(dii_results))
            ax[1].imshow(ih.plot_rgb(img_rescale))
            ax[2].imshow(ih.plot_rgb(img_arr[i]))
            fname_substring = extract_date_from_filename(images[i])
            print(f"Finished processing: {fname_substring}")
            plt.savefig(f"/home/zba21/Documents/ShallowLearn/Data/24_Reef/figure_{fname_substring}.pdf")
            final_paths.append(images[i])
        except:
            print("Arr failed: argh")
            continue
    pd.DataFrame(final_paths).to_csv("/home/zba21/Documents/ShallowLearn/Data/24_Reef/filenames.csv")
    np.save("/home/zba21/Documents/ShallowLearn/Data/24_Reef/dii_arr.npy", np.array(dii_arr))
    print("Everything worked")