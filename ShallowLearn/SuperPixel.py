from skimage.data import astronaut
from skimage.color import rgb2gray
from skimage.filters import sobel, threshold_multiotsu
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler



def felzenszwalb_segmentation(image, scale=100, sigma=0.5, min_size=50):
    segments_fz = felzenszwalb(image, scale=scale, sigma=sigma, min_size=min_size)
    return segments_fz

def slic_segmentation(image, n_segments=300, compactness=10, sigma=1):
    segments_slic = slic(image, n_segments=n_segments, compactness=compactness, sigma=sigma, start_label=1)
    return segments_slic

def quickshift_segmentation(image, kernel_size=9, max_dist=30, ratio=0.5):
    segments_quick = quickshift(image, kernel_size=kernel_size, max_dist=max_dist, ratio=ratio)
    return segments_quick

def watershed_segmentation(image, markers = 250, compactness=0.001):
    segments_watershed = watershed(image, markers, compactness=compactness)
    return segments_watershed

def multiotsu_thresholding(image, classes=3):
    thresholds = threshold_multiotsu(image, classes=classes)
    return thresholds



def pre_pca(X, num_components=None):
    # Normalize the data
    X_normalized = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    
    # Calculate the covariance matrix
    covariance_matrix = np.cov(X_normalized, rowvar=False)
    
    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    
    # Sort the eigenvalues and eigenvectors in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]
    
    # Select the top 'num_components' eigenvectors
    if num_components is not None and num_components < len(eigenvalues):
        eigenvectors = eigenvectors[:, :num_components]
    
    # Project the data onto the principal components
    X_pca = np.dot(X_normalized, eigenvectors)
    
    return X_pca



def extract_patches(segments):
    flattened_patches = []
    for segment_id in np.unique(segments):
        mask = segments == segment_id
        patch = img[mask]
        flattened_patches.append(patch.flatten())

    return flattened_patches

def pad_patches(flattened_patches, max_length = None):
    if max_length is None:
        max_length = max(len(patch) for patch in flattened_patches)
    padded_patches = np.array([np.pad(patch, (0, max_length - len(patch)), 'constant', constant_values=0)
                           for patch in flattened_patches])
    return padded_patches, max_length

def pca_segments(patches, pca = None):
    from sklearn.decomposition import PCA
    if pca is None:
        pca = PCA(n_components=2, random_state=42)
        pca.fit(patches)
    pca_features = pca.transform(patches)
    return pca_features, pca

def cluster_segments(pca_features):
    from sklearn.cluster import DBSCAN
    clustering = DBSCAN(eps = 110, min_samples=5).fit_predict(pca_features)
    return clustering

def visualise_patch_clusters(image, segments, clusters):
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches

    # Create a new figure for the overlaid visualization
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)  # Show the original image

    # Define a colormap and a normalization instance to map cluster labels to colors
    colormap = plt.cm.viridis
    normalize = mcolors.Normalize(vmin=clusters.min(), vmax=clusters.max())

    # Iterate over each unique segment/patch to draw it
    for segment_id, cluster_label in zip(np.unique(segments), clusters):
        mask = segments == segment_id
        color = colormap(normalize(cluster_label))  # Get the color for the current cluster

        # Create an RGBA image for the current segment with the cluster color
        colored_patch = np.zeros((img.shape[0], img.shape[1], 4))
        colored_patch[mask] = color  # Use the RGBA color directly

        # Overlay the colored, semi-transparent patch on the original image
        ax.imshow(colored_patch, alpha=0.5)  # Adjust alpha for transparency

    # Generate legend handles based on the unique clusters
    handles = [mpatches.Patch(color=colormap(normalize(i)), label=f'Cluster {i}') for i in np.unique(clusters)]
    ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')

    return fig, ax 

def extract_dii(segments, original_segments, image, deep, shallow):
    """Im so sorry"""


    try:
        cluster_1_pca = segments[clusters == deep, 0]
        cluster_7_pca = segments[clusters == shallow, 0]
    except IndexError as e:
        print(f"IndexError caught: {e}")

        # Assuming the error is because 'segments' has fewer elements along dimension 0
        # Check which dimension is smaller and pad accordingly
        if segments.shape[0] < clusters.shape[0]:
            # Calculate the difference in dimensions
            diff = clusters.shape[0] - segments.shape[0]
            
            # Pad the 'segments' array. np.pad can be adjusted based on how you wish to pad
            # Here, we're padding with zeros - consider if this makes sense for your application
            segments_padded = np.pad(segments, ((0, diff), (0, 0)), mode='constant', constant_values=0)
            
            # Attempt to extract the clusters again with the padded array
            cluster_1_pca = segments_padded[clusters == deep, 0]
            cluster_7_pca = segments_padded[clusters == shallow, 0]
        else:
            print("The issue might not be with dimensions mismatch. Please check the arrays' shapes.")
    percentile_90_cluster_1_value = np.percentile(cluster_1_pca, 30)
    print(percentile_90_cluster_1_value)
    percentile_10_cluster_7_value = np.percentile(cluster_7_pca, 85)
    
    rightmost_cluster_1_idx = np.abs(cluster_1_pca - percentile_90_cluster_1_value).argmin()
    leftmost_cluster_7_idx = np.abs(cluster_7_pca - percentile_10_cluster_7_value).argmin()
    print(rightmost_cluster_1_idx)

    segments_cluster_1 = np.where(clusters == deep)[0][rightmost_cluster_1_idx]
    segments_cluster_7 = np.where(clusters == shallow)[0][leftmost_cluster_7_idx]
    print(segments_cluster_1.shape)

    # segments_cluster_1 = np.where(clusters == 0)[0][segments_cluster_1]
    # segments_cluster_7 = np.where(clusters == 1)[0][segments_cluster_7]


    mask_deep = original_segments == segments_cluster_1
    mask_shallow = original_segments == segments_cluster_7
    print(mask_deep.shape)

    # Assuming images[IMG_NO] is a 3D array of shape (height, width, num_bands)
    num_bands = image.shape[2]

    # Initialize an array to store DII results with the same shape as your image
    dii_results = np.empty_like(image, dtype=np.float32)

    for band in range(num_bands):
        # Extract band data
        band_data = image[:,:,band]
        
        # Calculate average reflectance in deep and shallow areas for the current band
        R_deep_avg = np.mean(band_data[mask_deep])
        R_shallow_avg = np.mean(band_data[mask_shallow])
        print(R_deep_avg, R_shallow_avg)
        
        # Compute DII for each pixel in the current band
        # Use np.where to avoid taking the log of non-positive values
        R_diff = band_data - R_deep_avg
        dii_band = np.where(R_diff > 0, np.log(R_diff), 0)
        
        # Store the DII result for the current band
        dii_results[:,:,band] = dii_band
    return dii_results

def mask_out_zeros(image_array):
    """
    Masks out zeros in an image represented as a numpy array.
    Zeros are replaced with NaN. Adjust if a different replacement is needed.

    Parameters:
    - image_array: numpy array representing an image

    Returns:
    - masked_image: numpy array with zeros masked out
    """
    masked_image = np.where(image_array == 0, np.nan, image_array)
    return masked_image



def min_max_scaler(arr, scaler = None):
    if scaler is None:
        scaler = MinMaxScaler()
    arr_reshaped = arr.reshape(-1, 1)
    arr_scaled = scaler.fit_transform(arr_reshaped)
    return arr_scaled.reshape(arr.shape) 

if __name__ == "__main__":
    import ShallowLearn.LoadData as load_data
    import ShallowLearn.LabelLoader as label_loader
    import ShallowLearn.ImageHelper as ih
    import numpy as np
    import matplotlib.pyplot as plt
    import ShallowLearn.ComputerVisionFeatures as cvf
    import pandas as pd
    from ShallowLearn import band_mapping

    seasonal_loader = load_data.LoadSeasonalData("/mnt/sda_mount/Clipped/L1C/")
    dates = seasonal_loader.generate_dates()
    seasonal_images = seasonal_loader.gen_seasonal_images()
    # print(seasonal_images)
    season = 'Winter'
    print(np.array(seasonal_loader.load_seasonal_images(f"{season}")).shape)
    # fig, ax = plt.subplots(1,1, figsize=(10,10))

    images = seasonal_loader.load_seasonal_images(f"{season}")
    median_images = np.ma.median(images, axis = 0)
    median_images_filled = median_images.filled(np.nan)
    # plt.imshow(ih.plot_rgb(median_images_filled))
    # plt.show()

    IMG_NO = 1
    img = ih.plot_rgb(images[IMG_NO], plot=False)
    # ih.plot_rgb(images[IMG_NO], plot=True)
    # img = img_as_float(img)
    # plt.show()

    slic_segments = slic_segmentation(img, sigma = 1.0)
    # plt.imshow(mark_boundaries(img, slic_segments))
    # plt.show()
    # fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    extracted_patches = extract_patches(slic_segments)
    pcad_segments, max_length = pad_patches(extracted_patches)

    pcad_segments, pca = pca_segments(pcad_segments)
    print(pcad_segments)
    clusters = cluster_segments(pcad_segments)
    # plt.scatter(pcad_segments[:, 0], pcad_segments[:, 1], c=clusters, cmap='viridis')
    # plt.show()

    # visualise_patch_clusters(img, slic_segments, clusters)
    # plt.show()

    dii = (extract_dii(pcad_segments, slic_segments, img, 0, 1))
    print(dii.shape)
    # fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    # ax.imshow(min_max_scaler(dii))
    # plt.show()
    # fit kmeans and plot 
    import ShallowLearn.RadiometricNormalisation as rn 
    from skimage.morphology import disk, cube
    from skimage import exposure
    from skimage.filters import rank
    # footprint = cube(50)
    # dii = min_max_scaler(dii)

    # dii = rank.equalize(dii, footprint=footprint)

    from sklearn.cluster import KMeans
    from skimage.filters import threshold_multiotsu
    thresholds = threshold_multiotsu(dii[:,:,[2]])
    from sklearn.preprocessing import RobustScaler
    # Using the threshold values, we generate the three regions.
    regions = np.digitize(dii[:,:,[2]], bins=thresholds)
    # plt.colorbar()
    mask = regions == 2
    # dii, scaler = min_max_scaler(dii)
    N_CLUSTERS = 9

    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42)
    scaler = RobustScaler()
    kmeans.fit(scaler.fit_transform(ih.apply_mask(dii, mask).reshape(-1, 3)))
    segmented_img = kmeans.labels_
    segmented_img = segmented_img.reshape(dii.shape[:2])


    for i in range(2, len(images)):
        try:
            img_2 = ih.plot_rgb(images[i], plot=False)
            slic_segments_2 = slic_segmentation(img_2, sigma = 1.5)
            extracted_patches_2 = extract_patches(slic_segments_2)
            pcad_segments_2, _ = pad_patches(extracted_patches_2, max_length)
            pcad_segments_2, pca = pca_segments(pcad_segments_2, pca)
            clusters_2 = cluster_segments(pcad_segments_2)

            dii_2 = (extract_dii(pcad_segments_2, slic_segments_2, img_2, 0, 1))
            # dii_2 = min_max_scaler(dii_2)

            # dii_2 = rank.equalize(dii_2, footprint=footprint)
            # dii_2, scaler = min_max_scaler(dii_2, scaler)
            thresholds_2 = threshold_multiotsu(dii_2[:,:,[2]])

            # Using the threshold values, we generate the three regions.
            regions_2 = np.digitize(dii_2[:,:,[2]], bins=thresholds_2)
            # plt.colorbar()
            mask_2 = regions_2 == 2
            kmeans_2 =  KMeans(n_clusters = N_CLUSTERS,  init = kmeans.cluster_centers_, 
                                    random_state = 42, max_iter = 500)
            
            segmented_img_2 = kmeans_2.fit(scaler.transform(ih.apply_mask(dii_2, mask_2).reshape(-1, 3))).labels_
            print(kmeans_2.n_iter_)
            segmented_img_2 = segmented_img_2.reshape(dii_2.shape[:2])

            

            fig, ax = plt.subplots(2, 2, figsize=(10, 10))
            # make plots of the dii and segmented images
            ax[0, 0].imshow(ih.apply_mask(min_max_scaler(dii), mask))
            ax[0, 1].imshow(segmented_img)
            ax[1, 0].imshow(ih.apply_mask(min_max_scaler(dii_2), mask_2))
            ax[1, 1].imshow(segmented_img_2)
            plt.show()
        except:
            print("Error")
            pass
