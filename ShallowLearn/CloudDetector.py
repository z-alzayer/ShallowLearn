import xgboost as xgb
import numpy as np
import ShallowLearn.ImageHelper as ih
from scipy.ndimage import binary_dilation, binary_erosion


def detect_clouds(datacube, threshold = 0.0, window_size=8):
    """
    Detects clouds in a time series of images by comparing each pixel to the mean of the surrounding window.

    Parameters:
    - datacube: 4D numpy array (time, x, y, channels)
    - window_size: number of images in the sliding window

    Returns:
    - cloud_mask: 4D boolean numpy array (time, x, y, channels) indicating cloud presence for each image
    """
    datacube = np.array(datacube)
    time, x, y, channels = datacube.shape
    half_window = window_size // 2
    
    # Initialize cloud mask with the same shape as datacube
    cloud_mask = np.zeros((time, x, y, channels), dtype=bool)
    
    # Pad the datacube to handle the boundaries
    padded_datacube = np.pad(datacube, ((half_window, half_window), (0, 0), (0, 0), (0, 0)), mode='reflect')
    
    for t in range(time):
        window = padded_datacube[t:t + window_size]
        window_mean = np.mean(window, axis=0)
               
        cloud_mask[t] = datacube[t] > (window_mean  + threshold)
    
    return cloud_mask




def add_nan_buffer(arr, dilation_size=3):
    """
    Adds a buffer around NaN values in the given array using morphological dilation.

    Parameters:
        arr (np.ndarray): The input array.
        dilation_size (int): Size of the dilation structure. This controls the thickness of the NaN buffer.

    Returns:
        np.ndarray: The modified array with a buffer around NaN values.
    """


    # Define the structure for dilation based on the given dilation size
    structure = np.ones((dilation_size, dilation_size))

    # Perform binary dilation on the NaN mask
    dilated_mask = binary_erosion(arr, structure=structure)

    return dilated_mask

def cloud_regressor(img, return_mask = False, threshold = 500,
                    planet = False, dilation = True, dilation_size = 10,
                    model_path = "/home/zba21/Documents/ShallowLearn/Models/CloudDetectXGB.json", processed = False):
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    # img_copy = img.copy()
    if planet == False:
        img, shape, original = load_img_model(img, processed = processed)
    else:
        img, shape, original = load_img_planet(img)

                
    mask = np.expand_dims(model.predict(img).reshape(shape[:2]), axis = 2)
    print("Generating cloud mask complete.")
    # import matplotlib.pyplot as plt
    # plt.imshow(mask)
    # plt.show()

    # print(mask.shape)
    if return_mask == True:
        return mask
    
    if dilation:
        mask = mask < threshold
        mask = add_nan_buffer(mask[:,:,0], dilation_size = dilation_size)
        mask = np.expand_dims(mask, axis = 2)

    return ih.apply_mask(original, mask)

def percentile_without_zeros(arr, q):
    non_zero_arr = arr[arr != 0]
    if len(non_zero_arr) == 0:
        return np.nan
    return np.percentile(non_zero_arr, q)

def percentile_without_zeros_and_first(arr, q):
    non_zero_arr = arr[arr != 0]
    if len(non_zero_arr) == 0:
        return np.nan
    non_zero_arr = np.unique(np.sort(non_zero_arr))[1:]  # Remove repeated smallest values
    return np.percentile(non_zero_arr, q)


def load_img_model(img, processed  = False):
    if processed == True:
        return img.reshape(-1,4), img.shape, img 
    if isinstance(img, str):
        img = ih.load_img(img)

    return img[:,:,[4,3,2,8]].reshape(-1,4), img.shape, img


def load_img_planet(img, processed  = False):
    if processed == True:
        return img.reshape(-1,4), img.shape, img 
    if isinstance(img, str):
        img = ih.load_img(img)

    return img[:,:,[3,2,1,0]].reshape(-1,4), img.shape, img