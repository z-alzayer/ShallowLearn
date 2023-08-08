import pandas as pd 
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import ShallowLearn.Transform as tf
import math 

def create_scatter_cmap(df, cols = None, alpha = None):
    """converts colour space to a format compatible with matplotlib scatter
        of values between -1 and 1 then linearly converts it back to 0 and 1
          - using B08 as alpha, input columns are
        a list"""
    # make a copy of the dataframe and so that changes are not inplace
    df_copy = df.copy()

    if cols is not None and len(cols) > 4:
        raise ValueError("Too many columns, only 4 allowed")

    if cols is None:
        # We assume the alpha is the last channel
        cols = ["B04", "B03", "B02", "B08"]

    for column in cols:
        df_copy[column] = np.clip(df_copy[column], -1, 1)
    for column in cols:
        df_copy[column] = (df_copy[column] + 1) / 2

    if alpha is not None and len(cols) == 4:
        df_copy[cols[-1]] = df_copy[cols[-1]] * alpha

    df_copy['colour'] = list(zip(*[df_copy[col] for col in cols]))
    return df_copy['colour']

def reshape_to_square(image, N = 4):
    # Calculate the total number of elements divided by the channel size (4 in this case)
    total_elements = image.size // image.shape[-1]
    
    # Find n
    n = math.ceil(math.sqrt(total_elements))
    
    # Create a zero-filled reshaped_image
    reshaped_image = np.zeros((n, n, N))
    
    # Flatten the original image for easier indexing
    flat_image = image.reshape(-1, image.shape[-1])
    
    # Fill the reshaped_image
    for idx, row in enumerate(flat_image):
        i = idx // n
        j = idx % n
        reshaped_image[i, j] = row
        
    return reshaped_image


def sort_multidimensional_data(image):
    """
    Sorts a multidimensional array based on the average value of the last dimension.

    Parameters:
    - image (ndarray): A multidimensional numpy array.

    Returns:
    - ndarray: The image sorted by the average value of the last dimension.
    """

    # Compute the 'sorting criterion' as the mean along the last axis
    # Simple sorting method - if we sort by all the arrays its too ram intensive
    sorting_criterion = np.mean(image, axis=-1)

    # Get sorting indices from sorting criterion
    sorting_indices = np.argsort(sorting_criterion.ravel())

    # Flatten and sort the original image with these indices
    sorted_flattened = image.reshape(-1, image.shape[-1])[sorting_indices]

    # Reshape to get the sorted image
    sorted_data = sorted_flattened.reshape(image.shape)
    
    return sorted_data

if __name__ == "__main__":
    print("Hello")