import functools
import numpy as np

def remove_zeros_from_image(image):
    """
    Removes zeros from a multiband image and returns the mask.
    Assumes the image is a NumPy array.
    """
    # Create a mask where any of the bands is zero
    # Replace zeros with np.nan in each band
    image_with_nan = np.where(image == 0, np.nan, image)

    return image_with_nan

def remove_zeros_decorator(func):
    """
    Decorator that applies the remove_zeros_from_image function
    to the first argument of the decorated function.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Apply the mask to the first argument
        masked_image = remove_zeros_from_image(args[0])

        # Call the original function with the masked image
        return func(masked_image, *args[1:], **kwargs)

    return wrapper