import numpy as np

def extract_point_spectra(array, x, y):
    """
    Extracts the spectra of a single point (x, y) across all time slices.
    
    Parameters:
    - array: numpy array of shape (t, x, y, z)
    - x: x-coordinate of the point
    - y: y-coordinate of the point
    
    Returns:
    - spectra: numpy array of shape (t, z)
    """
    
    if len(array.shape) != 4:
        raise ValueError("Input array should be of shape (t, x, y, z)")
    
    spectra = array[:, x, y, :]
    return spectra