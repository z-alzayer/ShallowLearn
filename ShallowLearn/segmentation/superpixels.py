"""
Image segmentation utilities for superpixel generation.
Clean implementation without redundant code.
"""

import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
from skimage.filters import threshold_multiotsu
from skimage.segmentation import felzenszwalb, quickshift, slic, watershed
from skimage.transform import resize
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler

# Import StandardDII for depth invariant calculations
from ..features.standard_dii import apply_depth_invariant_index, calculate_slope_from_values

# Suppress sklearn warnings
warnings.filterwarnings("ignore", category=UserWarning)


def felzenszwalb_segmentation(
    image: np.ndarray, scale: float = 100, sigma: float = 0.5, min_size: int = 50
) -> np.ndarray:
    """
    Performs Felzenszwalb segmentation on an image.

    Parameters:
    -----------
    image : np.ndarray
        Input image array
    scale : float, default=100
        Segmentation scale parameter
    sigma : float, default=0.5
        Gaussian kernel standard deviation
    min_size : int, default=50
        Minimum segment size

    Returns:
    --------
    np.ndarray
        Segmentation labels
    """
    if image.shape[-1] != 3:
        image = image[:,:,[4,3,2]]
    return felzenszwalb(image, scale=scale, sigma=sigma, min_size=min_size)


def slic_segmentation(
    image: np.ndarray,
    n_segments: int = 1000,
    compactness: float = 10,
    sigma: float = 1,
    start_label: int = 1,
) -> np.ndarray:
    """
    Performs SLIC (Simple Linear Iterative Clustering) segmentation.

    Parameters:
    -----------
    image : np.ndarray
        Input image array
    n_segments : int, default=1000
        Approximate number of segments
    compactness : float, default=10
        Balances color proximity and space proximity
    sigma : float, default=1
        Gaussian smoothing parameter
    start_label : int, default=1
        Starting label for segments

    Returns:
    --------
    np.ndarray
        Segmentation labels
    """
    if image.shape[-1] != 3:
        image = image[:,:,[4,3,2]]
    return slic(
        image,
        n_segments=n_segments,
        compactness=compactness,
        sigma=sigma,
        start_label=start_label,
    )


def quickshift_segmentation(
    image: np.ndarray, kernel_size: float = 3, max_dist: float = 6, ratio: float = 0.5
) -> np.ndarray:
    """
    Performs quickshift segmentation.

    Parameters:
    -----------
    image : np.ndarray
        Input image array
    kernel_size : float, default=3
        Width of Gaussian kernel for density estimation
    max_dist : float, default=6
        Cut-off point for data distances
    ratio : float, default=0.5
        Balances color-space proximity and image-space proximity

    Returns:
    --------
    np.ndarray
        Segmentation labels
    """
    if image.shape[-1] != 3:
        image = image[:,:,[4,3,2]]
    return quickshift(image, kernel_size=kernel_size, max_dist=max_dist, ratio=ratio)


def watershed_segmentation(
    image: np.ndarray, markers: Optional[np.ndarray] = None, connectivity: int = 1
) -> np.ndarray:
    """
    Performs watershed segmentation.

    Parameters:
    -----------
    image : np.ndarray
        Input image array
    markers : np.ndarray, optional
        Markers for watershed
    connectivity : int, default=1
        Connectivity for watershed

    Returns:
    --------
    np.ndarray
        Segmentation labels
    """
    if markers is None:
        # Use gradient for watershed if no markers provided
        from skimage.filters import sobel

        elevation_map = sobel(image)
        markers = np.zeros_like(elevation_map, dtype=int)
        markers[elevation_map < 0.1] = 1
        markers[elevation_map > 0.8] = 2

    return watershed(image, markers, connectivity=connectivity)


def multiotsu_thresholding(
    image: np.ndarray, classes: int = 3, nbins: int = 256
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies multi-Otsu thresholding to generate multiple classes.

    Parameters:
    -----------
    image : np.ndarray
        Input grayscale image
    classes : int, default=3
        Number of classes for thresholding
    nbins : int, default=256
        Number of bins for histogram

    Returns:
    --------
    tuple
        (thresholds, segmented_image)
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        image = np.mean(image, axis=2)

    thresholds = threshold_multiotsu(image, classes=classes, nbins=nbins)
    segmented = np.digitize(image, bins=thresholds)

    return thresholds, segmented


def extract_patches(
    image: np.ndarray, segments: np.ndarray, min_size: int = 10
) -> List[np.ndarray]:
    """
    Extracts patches from image based on segmentation labels.

    Parameters:
    -----------
    image : np.ndarray
        Input image array
    segments : np.ndarray
        Segmentation labels
    min_size : int, default=10
        Minimum patch size to include

    Returns:
    --------
    List[np.ndarray]
        List of extracted patches
    """
    patches = []
    unique_labels = np.unique(segments)

    for label in unique_labels:
        if label == 0:  # Skip background
            continue

        mask = segments == label
        if np.sum(mask) < min_size:
            continue

        # Find bounding box
        rows, cols = np.where(mask)
        min_row, max_row = np.min(rows), np.max(rows) + 1
        min_col, max_col = np.min(cols), np.max(cols) + 1

        # Extract patch
        patch_mask = mask[min_row:max_row, min_col:max_col]
        if len(image.shape) == 3:
            patch = image[min_row:max_row, min_col:max_col, :]
            patch = patch * patch_mask[:, :, np.newaxis]
        else:
            patch = image[min_row:max_row, min_col:max_col]
            patch = patch * patch_mask

        patches.append(patch)

    return patches


def pca_segments(
    patches: List[np.ndarray], n_components: int = 5, return_transformer: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, PCA]]:
    """
    Applies PCA to flattened patches.

    Parameters:
    -----------
    patches : List[np.ndarray]
        List of patch arrays
    n_components : int, default=5
        Number of PCA components
    return_transformer : bool, default=False
        Whether to return the PCA transformer

    Returns:
    --------
    np.ndarray or tuple
        PCA-transformed data, optionally with transformer
    """
    # Flatten patches
    flattened_patches = []
    for patch in patches:
        if len(patch.shape) == 3:
            flattened = patch.reshape(-1, patch.shape[2])
        else:
            flattened = patch.reshape(-1, 1)

        # Remove zero pixels
        non_zero_mask = np.any(flattened != 0, axis=1)
        if np.sum(non_zero_mask) > 0:
            flattened_patches.append(flattened[non_zero_mask])

    if not flattened_patches:
        if return_transformer:
            return np.array([]), None
        return np.array([])

    # Concatenate all patches
    all_pixels = np.vstack(flattened_patches)

    # Apply PCA
    pca = PCA(n_components=min(n_components, all_pixels.shape[1]))
    transformed = pca.fit_transform(all_pixels)

    if return_transformer:
        return transformed, pca
    return transformed


def cluster_segments(
    features: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 5,
    algorithm: str = "auto",
) -> np.ndarray:
    """
    Clusters features using DBSCAN.

    Parameters:
    -----------
    features : np.ndarray
        Input features for clustering
    eps : float, default=0.5
        DBSCAN epsilon parameter
    min_samples : int, default=5
        DBSCAN minimum samples parameter
    algorithm : str, default='auto'
        DBSCAN algorithm choice

    Returns:
    --------
    np.ndarray
        Cluster labels
    """
    if features.size == 0:
        return np.array([])

    clustering = DBSCAN(eps=eps, min_samples=min_samples, algorithm=algorithm)
    labels = clustering.fit_predict(features)

    return labels


def scale_features(
    features: np.ndarray, method: str = "minmax", return_scaler: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, MinMaxScaler]]:
    """
    Scales features using various methods.

    Parameters:
    -----------
    features : np.ndarray
        Input features
    method : str, default='minmax'
        Scaling method ('minmax', 'standard', 'robust')
    return_scaler : bool, default=False
        Whether to return the scaler object

    Returns:
    --------
    np.ndarray or tuple
        Scaled features, optionally with scaler
    """
    if features.size == 0:
        if return_scaler:
            return features, None
        return features

    if method == "minmax":
        from sklearn.preprocessing import MinMaxScaler

        scaler = MinMaxScaler()
    elif method == "standard":
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
    elif method == "robust":
        from sklearn.preprocessing import RobustScaler

        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown scaling method: {method}")

    scaled = scaler.fit_transform(features)

    if return_scaler:
        return scaled, scaler
    return scaled


def extract_dii(
    image: np.ndarray,
    clusters: np.ndarray,
    segments: np.ndarray,
    patches: List[np.ndarray] = None,
    method: str = "mean",
) -> dict:
    """
    Extracts Digital Intertidal Index (DII) from clustered segments.

    Parameters:
    -----------
    image : np.ndarray
        Input image array
    clusters : np.ndarray
        Cluster labels for patches/pixels
    segments : np.ndarray
        Segmentation labels
    patches : List[np.ndarray], optional
        List of patches corresponding to cluster labels
    method : str, default='mean'
        Aggregation method ('mean', 'median', 'std')

    Returns:
    --------
    dict
        DII results by cluster
    """
    dii_results = {}
    unique_clusters = np.unique(clusters)

    # If no patches provided, extract them from segments
    if patches is None:
        patches = extract_patches(image, segments)

    # Ensure clusters array matches patches length
    if len(clusters) != len(patches):
        # If mismatch, compute DII directly from segments
        for cluster_id in unique_clusters:
            if cluster_id == -1:  # Skip noise cluster
                continue

            # Get representative segment for this cluster
            segment_id = cluster_id + 1  # Simple mapping
            if segment_id in np.unique(segments):
                seg_mask = segments == segment_id
                if len(image.shape) == 3:
                    pixels = image[seg_mask]
                else:
                    pixels = image[seg_mask].reshape(-1, 1)

                if len(pixels) > 0:
                    if method == "mean":
                        dii_value = np.mean(pixels, axis=0)
                    elif method == "median":
                        dii_value = np.median(pixels, axis=0)
                    elif method == "std":
                        dii_value = np.std(pixels, axis=0)
                    else:
                        raise ValueError(f"Unknown aggregation method: {method}")

                    dii_results[cluster_id] = dii_value
    else:
        # Use patch-based approach
        for cluster_id in unique_clusters:
            if cluster_id == -1:  # Skip noise cluster
                continue

            cluster_mask = clusters == cluster_id
            cluster_patches = [
                patches[i] for i in range(len(patches)) if cluster_mask[i]
            ]

            if cluster_patches:
                # Extract non-zero pixels from patches
                cluster_pixels = []
                for patch in cluster_patches:
                    if len(patch.shape) == 3:
                        pixels = patch.reshape(-1, patch.shape[2])
                    else:
                        pixels = patch.reshape(-1, 1)

                    # Remove zero pixels
                    non_zero_mask = (
                        np.any(pixels != 0, axis=1)
                        if len(pixels.shape) > 1
                        else pixels != 0
                    )
                    if np.sum(non_zero_mask) > 0:
                        cluster_pixels.append(pixels[non_zero_mask])

                if cluster_pixels:
                    all_pixels = np.vstack(cluster_pixels)

                    if method == "mean":
                        dii_value = np.mean(all_pixels, axis=0)
                    elif method == "median":
                        dii_value = np.median(all_pixels, axis=0)
                    elif method == "std":
                        dii_value = np.std(all_pixels, axis=0)
                    else:
                        raise ValueError(f"Unknown aggregation method: {method}")

                    dii_results[cluster_id] = dii_value

    return dii_results


def create_superpixel_dii_stack(
    image: np.ndarray,
    n_segments: int = 110,
    bands: List[int] = [0, 1, 2],
    correction_factor: int = 10,
    segmentation_method: str = "slic",
    band_combos=None,
    method: str = 'linear',
    method_kwargs: dict = None) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Create superpixels and generate DII stack following StandardDII methodology.

    Parameters:
    -----------
    image : np.ndarray
        Input image array (height, width, channels)
    n_segments : int, default=110
        Number of superpixel segments
    bands : List[int], default=[0, 1, 2]
        Bands to use for superpixel creation
    correction_factor : int, default=10
        Compactness factor for SLIC segmentation
    segmentation_method : str, default='slic'
        Segmentation method to use
    band_combos : List[Tuple[int, int]], optional
        Band combinations for DII calculation
    method : str, default='linear'
        Regression method for DII calculation
    method_kwargs : dict, optional
        Additional keyword arguments for the chosen method

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, dict]
        (features, segments, results_dict)
    """
    # Initialize method_kwargs if None
    if method_kwargs is None:
        method_kwargs = {}
    
    # Create superpixel segmentation
    if segmentation_method == "slic":
        segments = slic_segmentation(
            image, n_segments=n_segments, compactness=correction_factor
        )
    elif segmentation_method == "felzenszwalb":
        segments = felzenszwalb_segmentation(image)
    elif segmentation_method == "quickshift":
        segments = quickshift_segmentation(image)
    else:
        raise ValueError(f"Unknown segmentation method: {segmentation_method}")

    # Process with DII pipeline
    results = process_superpixel_dii_pipeline(
        image, segments, bands=bands, band_combos=band_combos, 
        method=method, method_kwargs=method_kwargs
    )

    # Debug: Check what results is
    print(f"Type of results: {type(results)}")
    print(f"Results keys: {results.keys() if isinstance(results, dict) else 'Not a dict'}")

    # Extract features (mean values for each superpixel)
    features = results["features"]

    return features, segments, results


def process_superpixel_dii_pipeline(
    image: np.ndarray,
    segments: np.ndarray,
    bands: List[int] = [0, 1, 2],
    n_components: int = 3,
    band_combos: List[Tuple[int, int]] = None,
    method: str = 'linear',
    method_kwargs: dict = None) -> dict:
    """
    Complete superpixel processing pipeline using StandardDII approach.

    Parameters:
    -----------
    image : np.ndarray
        Input image array (height, width, channels)
    segments : np.ndarray
        Segmentation labels array
    bands : List[int], default=[0, 1, 2]
        Bands to use for PCA transformation
    n_components : int, default=3
        Number of components for Gaussian Mixture Model
    band_combos : List[Tuple[int, int]], optional
        Band combinations for DII calculation. If None, uses default combinations.
    method : str
        Regression method for DII calculation
        Options: 'linear', 'polynomial', 'power_law', 'spline'
        Default: 'linear'
    method_kwargs : dict, optional
        Additional keyword arguments for the chosen method:
        - For 'polynomial': {'poly_degree': int} (default: 2)
        - For 'spline': {'smoothing': float, 'spline_k': int} (defaults: None, 3)
        - For 'power_law': no additional parameters needed
        Examples:
            {'poly_degree': 3}
            {'smoothing': 0.1, 'spline_k': 3}

    Returns:
    --------
    dict
        Results containing segments, deep/shallow masks, clusters, and DII stack
    """
    if method_kwargs is None:
        method_kwargs = {}
        
    # Default band combinations for DII calculation
    if band_combos is None:
        band_combos = [
            (1, 5),
            (2, 3),
            (2, 4),
            (1, 2),
            (0, 3),
            (1, 4),
            (1, 3),
            (4, 8),
            (2, 9),
            (2, 5),
        ]

    # Extract superpixel features for specified bands
    unique_segments = np.unique(segments)
    if unique_segments[0] == 0:  # Remove background if present
        unique_segments = unique_segments[1:]

    # Create feature matrix: each row is a superpixel, columns are band values
    features = []
    for segment_id in unique_segments:
        mask = segments == segment_id
        segment_pixels = image[mask]
        if len(segment_pixels) > 0:
            # Use mean values for the specified bands
            segment_features = np.mean(segment_pixels[:, bands], axis=0)
            features.append(segment_features)
        else:
            features.append(np.zeros(len(bands)))

    features = np.array(features)

    # Apply PCA transformation
    pca = PCA(n_components=min(3, features.shape[1]))
    transformed = pca.fit_transform(features)

    # Apply Gaussian Mixture Model clustering
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    cluster_labels = gmm.fit_predict(transformed)

    # Determine the 'deep' cluster based on minimum mean value of first component
    deep_idx = np.argmin(gmm.means_[:, 0])

    # Create cluster map
    temp_arr = np.zeros_like(segments)
    for idx, segment_id in enumerate(unique_segments):
        temp_arr[segments == segment_id] = cluster_labels[idx]

    # Create deep and shallow masks
    deep_mask = temp_arr == deep_idx
    shallow_mask = temp_arr != deep_idx

    # Extract deep and shallow pixel values
    deep_pixels = image[deep_mask]
    shallow_pixels = image[shallow_mask]

    # Calculate DII for each band combination
    stack_shape = (*image.shape[:2], len(band_combos))
    dii_stack = np.zeros(stack_shape)

    for idx, (band1, band2) in enumerate(band_combos):
        # Check if bands exist in image
        if band1 >= image.shape[2] or band2 >= image.shape[2]:
            continue

        # Calculate model using the specified method
        model, Ls = calculate_slope_from_values(
            deep_i=deep_pixels[:, band1],
            deep_j=deep_pixels[:, band2],
            shallow_i=shallow_pixels[:, band1],
            shallow_j=shallow_pixels[:, band2],
            method=method,
            **method_kwargs
        )

        # Apply DII transformation to entire image
        dii_stack[:, :, idx] = apply_depth_invariant_index(
            image[:, :, band1], 
            image[:, :, band2], 
            model,
            Ls,
            method=method
        )

    result_dict = {
        "segments": segments,
        "cluster_map": temp_arr,
        "deep_mask": deep_mask,
        "shallow_mask": shallow_mask,
        "deep_idx": deep_idx,
        "cluster_labels": cluster_labels,
        "features": features,
        "transformed_features": transformed,
        "gmm": gmm,
        "pca": pca,
        "dii_stack": dii_stack,
        "band_combos": band_combos,
    }
    
    print(f"About to return dict with keys: {result_dict.keys()}")
    return result_dict
def pad_slice_segments(image: np.ndarray, segments: np.ndarray, shape: Tuple[int, int, int] = (32, 32, 3)) -> np.ndarray:
    """
    Extract and resize patches from image based on segments.
    
    Parameters:
    -----------
    image : np.ndarray
        Input image array
    segments : np.ndarray
        Segmentation labels
    shape : Tuple[int, int, int]
        Target shape for patches
        
    Returns:
    --------
    np.ndarray
        Array of resized patches
    """
    patches = []
    for i in np.unique(segments):
        segment = segments == i
        patch = image[segment]
        resized_patch = resize(patch, shape, preserve_range=True)
        patches.append(resized_patch)
    return np.array(patches)


def pad_slice_segments_w_0pads(image: np.ndarray, segments: np.ndarray, shape: Tuple[int, int, int] = (32, 32, 13)) -> np.ndarray:
    """
    Extract patches from image based on segments with zero padding.
    
    Parameters:
    -----------
    image : np.ndarray
        Input image array
    segments : np.ndarray
        Segmentation labels
    shape : Tuple[int, int, int]
        Target shape for patches
        
    Returns:
    --------
    np.ndarray
        Array of padded patches
    """
    patches = []
    for i in np.unique(segments):
        segment = segments == i
        patch = image[segment]
        
        # Calculate padding needed
        pad_size = shape[0] - len(patch)
        if pad_size > 0:
            # Create zero padding
            padding = np.zeros((pad_size,) + patch.shape[1:])
            # Concatenate original patch with padding
            padded_patch = np.concatenate([patch, padding], axis=0)
            patches.append(padded_patch)
        else:
            # If patch is larger than desired shape, resize it
            resized_patch = resize(patch, shape, preserve_range=True)
            patches.append(resized_patch)
    
    return np.array(patches)


def optics_labels(image: np.ndarray, segments: np.ndarray, min_samples: int = 10) -> np.ndarray:
    """
    Apply DBSCAN clustering to PCA-transformed superpixel patches.
    
    Parameters:
    -----------
    image : np.ndarray
        Input image array
    segments : np.ndarray
        Segmentation labels
    min_samples : int
        Minimum samples for DBSCAN
        
    Returns:
    --------
    np.ndarray
        Cluster labels
    """
    patches = pad_slice_segments(image, segments)
    pca_image = pca_segments(patches)
    db = DBSCAN(eps=10, min_samples=min_samples).fit(pca_image)
    return db.labels_


def generate_sup_pixel_labels(image: np.ndarray, no_segments: Optional[int] = None) -> np.ndarray:
    """
    Generate superpixel segmentation and labels.
    
    Parameters:
    -----------
    image : np.ndarray
        Input image array
    no_segments : Optional[int]
        Number of segments (auto-calculated if None)
        
    Returns:
    --------
    np.ndarray
        Segmentation labels
    """
    if no_segments is None:
        no_segments = int(np.sqrt(image.shape[0] * image.shape[1]) / 2)
    
    segments = slic_segmentation(image, n_segments=no_segments)
    return segments
