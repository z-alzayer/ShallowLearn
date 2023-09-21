import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import linregress

from ShallowLearn.Transform import BCET, linear_contrast_enhancement, LCE_multi
import matplotlib.pyplot as plt


from skimage import exposure

def histogram_matching(source_img, reference_img):
    """
    Match the histogram of the source_img to the histogram of the reference_img.
    """
    matched_img = np.empty_like(source_img)
    for band in range(source_img.shape[-1]):
        matched_img[..., band] = exposure.match_histograms(source_img[..., band], reference_img[..., band])
    return matched_img


def remove_zeros_from_pair(source_band, reference_band):
    """
    Masks zeros from source_band and reference_band.
    Returns arrays without zero values.
    """
    mask = (source_band != 0) & (reference_band != 0)
    return source_band[mask], reference_band[mask]


def pca_filter_and_normalize(source_img, reference_img, threshold=1.0):
    """
    Radiometric normalization based on a reference image using PCA, 
    but only for the pixels that are close to the major principal component.

    Parameters:
    - source_img : numpy array
        Source image to be normalized.
    - reference_img : numpy array
        Reference image.
    - threshold : float (optional)
        Threshold for filtering pixels based on their position relative to the major principal component. Default is 1.0.

    Returns:
    - numpy array : Normalized source image.
    """
    
    # Ensure source and reference images have the same shape
    assert source_img.shape == reference_img.shape, "The source and reference images must have the same shape."

    # Create an empty array for the normalized source image with the same shape as the source_img
    normalized_img = np.empty_like(source_img)
    # source_img = histogram_matching(source_img, reference_img)
    # Process each band
    for band in range(source_img.shape[-1]):
        src_band = source_img[..., band].ravel()
        ref_band = reference_img[..., band].ravel()

        # Create a joint dataset
        joint_data = np.vstack((src_band, ref_band)).T

        # Apply PCA on the joint dataset
        pca = PCA(n_components=2)
        pca.fit(joint_data)

        # Transform the joint data to the PCA space
        transformed_data = pca.transform(joint_data)

        # Extract values in the direction of the major eigenvector
        major_values = transformed_data[:, 1]

        # Filter based on the threshold
        valid_indices = np.logical_and(major_values >= (-threshold), major_values <= threshold)

        # Perform normalization on the valid pixels
        slope, intercept, _, _, _ = linregress(src_band[valid_indices], ref_band[valid_indices])
        normalized_band = linear_contrast_enhancement(src_band * slope + intercept).astype(source_img.dtype).reshape(source_img[..., band].shape)
        print("Band: {}, Slope: {}, Intercept: {}".format(band, slope, intercept))
        # Assign the normalized band to the output image
        normalized_img[..., band] = normalized_band

    return normalized_img

def pca_filter_and_normalize_b8(source_img, reference_img,band_8, threshold=1.0):
    """
    Radiometric normalization based on a reference image using PCA, 
    but only for the pixels that are close to the major principal component.

    Parameters:
    - source_img : numpy array
        Source image to be normalized.
    - reference_img : numpy array
        Reference image.
    - threshold : float (optional)
        Threshold for filtering pixels based on their position relative to the major principal component. Default is 1.0.

    Returns:
    - numpy array : Normalized source image.
    """
    
    # Ensure source and reference images have the same shape
    assert source_img.shape == reference_img.shape, "The source and reference images must have the same shape."

    # Create an empty array for the normalized source image with the same shape as the source_img
    normalized_img = np.empty_like(source_img)
    # source_img = histogram_matching(source_img, reference_img)
    # Process each band
    src_b8 = source_img[..., band_8].ravel()
    ref_b8 = reference_img[..., band_8].ravel()

    for band in range(source_img.shape[-1]):
        src_band = source_img[..., [band]].ravel()
        ref_band = reference_img[..., [band]].ravel()
        src_band = src_band / (src_b8 + 1)
        ref_band = ref_band / (ref_b8 + 1)
        # Create a joint dataset
        joint_data = np.vstack((src_band, ref_band)).T
        joint_data = np.nan_to_num(joint_data)

        # Apply PCA on the joint dataset
        pca = PCA(n_components=2)
        pca.fit(joint_data)

        # Transform the joint data to the PCA space
        transformed_data = pca.transform(joint_data)

        # Extract values in the direction of the major eigenvector
        major_values = transformed_data[:, 1]

        # Filter based on the threshold
        valid_indices = np.logical_and(major_values >= (-threshold), major_values <= threshold)

        # Perform normalization on the valid pixels
        slope, intercept, _, _, _ = linregress(src_band[valid_indices], ref_band[valid_indices])
        normalized_band = linear_contrast_enhancement(src_band * slope + intercept).astype(source_img.dtype).reshape(source_img[..., band].shape)
        print("Band: {}, Slope: {}, Intercept: {}".format(band, slope, intercept))
        # Assign the normalized band to the output image
        normalized_img[..., band] = normalized_band

    return normalized_img



def pca_based_normalization(source_img, reference_img):
    """
    Radiometric normalization of a source image based on a reference image using PCA.
    The function takes in two numpy arrays (source_img and reference_img) and returns the normalized source image.
    """
    # Ensure source and reference images have the same shape
    assert source_img.shape == reference_img.shape, "The source and reference images must have the same shape."

    # Create an empty array for the normalized source image with the same shape as the source_img
    normalized_img = np.empty_like(source_img)
    source_img = histogram_matching(source_img, reference_img)
    # Assuming X number of bands, all bands are processed
    for band in range(source_img.shape[-1]):
        src_band = source_img[..., band]
        ref_band = reference_img[..., band]

        # Mask and remove zeros
        src_band_non_zero, ref_band_non_zero = remove_zeros_from_pair(src_band, ref_band)

        # Create a joint dataset
        joint_data = np.vstack((src_band_non_zero, ref_band_non_zero)).T

        # Apply PCA on the joint dataset
        pca = PCA(n_components=1,svd_solver = 'full')
        pca.fit(joint_data)

        # Extract first principal component (PC1)
        pc1 = pca.transform(joint_data)

        # Use PC1 to perform linear regression between source and reference
        slope, intercept, r_value, p_value, std_err = linregress(pc1[:, 0], joint_data[:, 1])

        print("Band: {}, Slope: {}, Intercept: {}".format(band, slope, intercept))

        # Apply the derived slope (gain) and intercept (offset) to normalize the source band
        normalized_data = linear_contrast_enhancement(src_band.ravel() * slope + intercept).astype(source_img.dtype)
        normalized_img[..., band] = normalized_data.reshape(src_band.shape)

    return normalized_img
# Example usage:
# source_image_np and reference_image_np are numpy arrays of the source and reference images

if __name__ == "__main__":
    from ShallowLearn.ImageHelper import plot_rgb
    images = np.load("/media/ziad/Expansion/Cleaned_Data_Directory/imgs.npy")
    # reference_image_np = np.load("/media/ziad/Expansion/Cleaned_Data_Directory/imgs.npy")
    image_ref_no = 30
    # ax[0].imshow(plot_rgb(images[0]))
    # ax[1].imshow(plot_rgb(images[image_ref_no]))
    # plt.show()

    ref = images[0]
    src = images[image_ref_no]
    hist_matched = histogram_matching(src, ref)
    normalized_image_np = pca_filter_and_normalize(src, ref)
    fig, ax = plt.subplots(1, 5, figsize=(15, 5))
    ax[0].imshow(plot_rgb(src))
    ax[1].imshow(plot_rgb(ref))
    ax[2].imshow(plot_rgb(normalized_image_np))
    ax[3].imshow(plot_rgb(hist_matched))
    ax[4].imshow(LCE_multi(plot_rgb(normalized_image_np - hist_matched)))

    ax[0].set_title("Source image")
    ax[1].set_title("Reference image")
    ax[2].set_title("PIF Normalized image")
    ax[3].set_title("Histogram matched image")
    ax[4].set_title("Difference")
    plt.show()
