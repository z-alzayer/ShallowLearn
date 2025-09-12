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

def histogram_matching_with_plot(source_img, reference_img):
    """
    Match the histogram of the source_img to the histogram of the reference_img.
    Returns the matched image.
    """
    matched_img = np.empty_like(source_img)
    num_bands = source_img.shape[-1]

    for band in range(num_bands):
        matched_img[..., band] = exposure.match_histograms(source_img[..., band], reference_img[..., band])

        # Plot the histograms
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.hist(source_img[..., band].ravel(), bins=256, color='blue', alpha=0.7, label='Source Histogram')
        plt.hist(reference_img[..., band].ravel(), bins=256, color='green', alpha=0.7, label='Reference Histogram')
        plt.title('Original Histograms - Band {}'.format(band))
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.hist(matched_img[..., band].ravel(), bins=256, color='red', alpha=0.7, label='Matched Histogram')
        plt.title('Matched Histogram - Band {}'.format(band))
        plt.legend()

        plt.subplot(1, 3, 3)
        plt.imshow(matched_img[..., band], cmap='gray')
        plt.title('Matched Image - Band {}'.format(band))

        plt.tight_layout()
        plt.show()

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
    # source_img = histogram_matching(source_img, reference_img)
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
# 
        # print("Band: {}, Slope: {}, Intercept: {}".format(band, slope, intercept))

        # Apply the derived slope (gain) and intercept (offset) to normalize the source band
        normalized_data = linear_contrast_enhancement(src_band.ravel() * slope + intercept).astype(source_img.dtype)
        normalized_img[..., band] = normalized_data.reshape(src_band.shape)

    return normalized_img


def pca_based_normalization_with_plot(source_img, reference_img):
    assert source_img.shape == reference_img.shape, "The source and reference images must have the same shape."

    normalized_img = np.empty_like(source_img)
    fig_index = 1
    num_bands = source_img.shape[-1]

    for band in range(num_bands):
        src_band = source_img[..., band]
        ref_band = reference_img[..., band]

        # Original source and reference plots
        plt.figure(fig_index)
        plt.subplot(1, 2, 1)
        plt.imshow(src_band, cmap='gray')
        plt.title('Original Source - Band {}'.format(band))

        plt.subplot(1, 2, 2)
        plt.imshow(ref_band, cmap='gray')
        plt.title('Reference - Band {}'.format(band))
        fig_index += 1

        src_band_non_zero, ref_band_non_zero = remove_zeros_from_pair(src_band.ravel(), ref_band.ravel())
        joint_data = np.vstack((src_band_non_zero, ref_band_non_zero)).T

        # Plot joint dataset in original space
        plt.figure(fig_index)
        plt.scatter(joint_data[:, 0], joint_data[:, 1], s=1)
        plt.title('Joint Data - Band {}'.format(band))
        plt.xlabel('Source Image Band')
        plt.ylabel('Reference Image Band')
        fig_index += 1

        pca = PCA(n_components=1, svd_solver='full')
        pca.fit(joint_data)
        pc1 = pca.transform(joint_data)

        # Plot PC1 vs reference
        plt.figure(fig_index)
        plt.scatter(pc1[:, 0], joint_data[:, 1], s=1)
        plt.title('PC1 vs Reference Image - Band {}'.format(band))
        plt.xlabel('PC1')
        plt.ylabel('Reference Image Band')
        fig_index += 1

        slope, intercept, _, _, _ = linregress(pc1[:, 0], joint_data[:, 1])
        print("Band: {}, Slope: {}, Intercept: {}".format(band, slope, intercept))
        normalized_data = linear_contrast_enhancement(src_band.ravel() * slope + intercept).astype(source_img.dtype)
        normalized_img[..., band] = normalized_data.reshape(src_band.shape)

        # Plot normalized image
        plt.figure(fig_index)
        plt.imshow(normalized_img[..., band], cmap='gray')
        plt.title('Normalized Source - Band {}'.format(band))
        fig_index += 1

    plt.show()

    return normalized_img

def pca_based_normalization_with_points(source_img, reference_img):
    """
    Radiometric normalization of a source image based on a reference image using PCA.
    Returns the normalized source image and an array marking the points used for PCA.
    """
    assert source_img.shape == reference_img.shape, "The source and reference images must have the same shape."
    normalized_img = np.empty_like(source_img)
    
    # Create an array to mark points used for PCA
    points_used_for_pca = np.zeros(source_img.shape[:2], dtype=bool)
    
    for band in range(source_img.shape[-1]):
        src_band = source_img[..., band]
        ref_band = reference_img[..., band]

        # Mask and remove zeros
        src_band_non_zero, ref_band_non_zero = remove_zeros_from_pair(src_band, ref_band)
        
        # Mark the points used for PCA in the mask
        mask = (src_band > 0) & (ref_band > 0)
        if band == 4:

            points_used_for_pca |= mask  # Update mask to include current band's points

        # Create a joint dataset
        joint_data = np.vstack((src_band_non_zero, ref_band_non_zero)).T

        # Apply PCA on the joint dataset
        pca = PCA(n_components=1, svd_solver='full')
        pca.fit(joint_data)

        # Extract first principal component (PC1)
        pc1 = pca.transform(joint_data)

        # Use PC1 to perform linear regression between source and reference
        slope, intercept, r_value, p_value, std_err = linregress(pc1[:, 0], joint_data[:, 1])

        print("Band: {}, Slope: {}, Intercept: {}".format(band, slope, intercept))

        # Normalize the source band
        normalized_data = (src_band.ravel() * slope + intercept).astype(source_img.dtype)
        normalized_img[..., band] = normalized_data.reshape(src_band.shape)

    return normalized_img, points_used_for_pca

def pca_filter_and_normalize_with_plot(source_img, reference_img, threshold=1.0):
    
    assert source_img.shape == reference_img.shape, "The source and reference images must have the same shape."
    normalized_img = np.empty_like(source_img)

    fig_index = 1
    num_bands = source_img.shape[-1]

    # Original plots
    for band in range(num_bands):
        plt.figure(fig_index)
        plt.subplot(1, 2, 1)
        plt.imshow(source_img[..., band], cmap='gray')
        plt.title('Original Source Image - Band {}'.format(band))
        
        plt.subplot(1, 2, 2)
        plt.imshow(reference_img[..., band], cmap='gray')
        plt.title('Reference Image - Band {}'.format(band))
        fig_index += 1

    for band in range(num_bands):
        src_band = source_img[..., band].ravel()
        ref_band = reference_img[..., band].ravel()

        joint_data = np.vstack((src_band, ref_band)).T
        pca = PCA(n_components=2)
        pca.fit(joint_data)
        transformed_data = pca.transform(joint_data)
        major_values = transformed_data[:, 1]
        valid_indices = np.logical_and(major_values >= (-threshold), major_values <= threshold)

        slope, intercept, _, _, _ = linregress(src_band[valid_indices], ref_band[valid_indices])
        normalized_band = linear_contrast_enhancement(src_band * slope + intercept).astype(source_img.dtype).reshape(source_img[..., band].shape)
        print("Band: {}, Slope: {}, Intercept: {}".format(band, slope, intercept))
        normalized_img[..., band] = normalized_band

        plt.figure(fig_index)
        plt.scatter(src_band, ref_band, s=1, c='blue', label='All Pixels')
        plt.scatter(src_band[valid_indices], ref_band[valid_indices], s=1, c='red', label='Valid Pixels')
        plt.title('Joint Data with Valid Pixels Highlighted - Band {}'.format(band))
        plt.xlabel('Source Image')
        plt.ylabel('Reference Image')
        plt.legend()
        fig_index += 1

        plt.figure(fig_index)
        plt.scatter(transformed_data[:, 0], transformed_data[:, 1], s=1, c='gray', label='All Pixels')
        plt.scatter(transformed_data[valid_indices, 0], transformed_data[valid_indices, 1], s=1, c='yellow', label='Valid Pixels')
        plt.axhline(y=threshold, color='g', linestyle='-')
        plt.axhline(y=-threshold, color='g', linestyle='-')
        plt.title('PCA Space with Valid Pixels Highlighted - Band {}'.format(band))
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.legend()
        fig_index += 1

        plt.figure(fig_index)
        plt.scatter(src_band[valid_indices], ref_band[valid_indices], s=1, c='cyan')
        x_vals = np.array(plt.gca().get_xlim())
        y_vals = intercept + slope * x_vals
        plt.plot(x_vals, y_vals, '--', color='black')
        plt.title('Linear Regression on Valid Pixels - Band {}'.format(band))
        plt.xlabel('Filtered Source Image')
        plt.ylabel('Reference Image')
        fig_index += 1

        plt.figure(fig_index)
        plt.imshow(normalized_img[..., band], cmap='gray')
        plt.title('Normalized Image - Band {}'.format(band))
        fig_index += 1

    plt.show()
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
