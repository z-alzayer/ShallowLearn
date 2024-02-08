import satellite_cloud_generator as scg
from functools import wraps
import numpy as np

import ShallowLearn.Transform as tf





def transpose_wrapper(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Assuming the first argument is the image
        image = args[0]

        # Transpose the image before passing it to the function
        transposed_image = np.swapaxes(tf.LCE_multi(image)/255, 0, 2)

        # Call the original function
        cloudy_img, cmask  = func(transposed_image, *args[1:], **kwargs)

        # Transpose the result
        cloudy_img = np.swapaxes(np.squeeze(cloudy_img), 0, 2)
        cmask = np.swapaxes(np.squeeze(cmask), 0, 2)

        return cloudy_img, cmask

    return wrapper



@transpose_wrapper
def gen_thin_clouds(img):
    """Takes in a normal image and returns a cloudy image and the cloud mask
        - the cloud method is a torch method so it needs to be converted to numpy
        current work around includes a cpu call and image is scaled to 0-1 as a side effect
        from the way the original library works"""
    
    cloudy_img, cmask = scg.add_cloud(img, return_cloud = True, min_lvl=0.1, max_lvl = .4)
    cloudy_img = cloudy_img.cpu().numpy()
    cmask = cmask.cpu().numpy()

    return cloudy_img, cmask

@transpose_wrapper
def gen_thick_clouds(img):
    """Takes in a normal image and returns a cloudy image and the cloud mask
        - the cloud method is a torch method so it needs to be converted to numpy
        current work around includes a cpu call and image is scaled to 0-1 as a side effect
        from the way the original library works"""
    
    cloudy_img, cmask = scg.add_cloud(img, return_cloud = True, min_lvl=0.4, max_lvl = .9)
    cloudy_img = cloudy_img.cpu().numpy()
    cmask = cmask.cpu().numpy()

    return cloudy_img, cmask
