"""
Image segmentation utilities for superpixel generation.
Clean implementation without redundant code.
"""
import numpy as np
from skimage.segmentation import (
    felzenszwalb, slic, quickshift, watershed
)
from skimage.filters import threshold_multiotsu
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List, Optional, Union
import warnings

# Suppress sklearn warnings
warnings.filterwarnings('ignore', category=UserWarning)


def felzenszwalb_segmentation(image: np.ndarray, 
                            scale: float = 100, 
                            sigma: float = 0.5, 
                            min_size: int = 50) -> np.ndarray:
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
    return felzenszwalb(image, scale=scale, sigma=sigma, min_size=min_size)


def slic_segmentation(image: np.ndarray, 
                     n_segments: int = 1000, 
                     compactness: float = 10, 
                     sigma: float = 1,
                     start_label: int = 1) -> np.ndarray:
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
    return slic(image, n_segments=n_segments, compactness=compactness, 
                sigma=sigma, start_label=start_label)


def quickshift_segmentation(image: np.ndarray, 
                          kernel_size: float = 3, 
                          max_dist: float = 6, 
                          ratio: float = 0.5) -> np.ndarray:
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
    return quickshift(image, kernel_size=kernel_size, max_dist=max_dist, ratio=ratio)


def watershed_segmentation(image: np.ndarray, 
                         markers: Optional[np.ndarray] = None,
                         connectivity: int = 1) -> np.ndarray:
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


def multiotsu_thresholding(image: np.ndarray, 
                          classes: int = 3,
                          nbins: int = 256) -> Tuple[np.ndarray, np.ndarray]:
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


def extract_patches(image: np.ndarray, 
                   segments: np.ndarray,
                   min_size: int = 10) -> List[np.ndarray]:
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


def pca_segments(patches: List[np.ndarray], 
                n_components: int = 5,
                return_transformer: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, PCA]]:
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


def cluster_segments(features: np.ndarray, 
                    eps: float = 0.5, 
                    min_samples: int = 5,
                    algorithm: str = 'auto') -> np.ndarray:
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


def scale_features(features: np.ndarray, 
                  method: str = 'minmax',
                  return_scaler: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, MinMaxScaler]]:
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
    
    if method == 'minmax':
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
    elif method == 'standard':
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
    elif method == 'robust':
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown scaling method: {method}")
    
    scaled = scaler.fit_transform(features)
    
    if return_scaler:
        return scaled, scaler
    return scaled


def extract_dii(image: np.ndarray, 
               clusters: np.ndarray,
               segments: np.ndarray,
               patches: List[np.ndarray] = None,
               method: str = 'mean') -> dict:
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
                    if method == 'mean':
                        dii_value = np.mean(pixels, axis=0)
                    elif method == 'median':
                        dii_value = np.median(pixels, axis=0)
                    elif method == 'std':
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
            cluster_patches = [patches[i] for i in range(len(patches)) if cluster_mask[i]]
            
            if cluster_patches:
                # Extract non-zero pixels from patches
                cluster_pixels = []
                for patch in cluster_patches:
                    if len(patch.shape) == 3:
                        pixels = patch.reshape(-1, patch.shape[2])
                    else:
                        pixels = patch.reshape(-1, 1)
                    
                    # Remove zero pixels
                    non_zero_mask = np.any(pixels != 0, axis=1) if len(pixels.shape) > 1 else pixels != 0
                    if np.sum(non_zero_mask) > 0:
                        cluster_pixels.append(pixels[non_zero_mask])
                
                if cluster_pixels:
                    all_pixels = np.vstack(cluster_pixels)
                    
                    if method == 'mean':
                        dii_value = np.mean(all_pixels, axis=0)
                    elif method == 'median':
                        dii_value = np.median(all_pixels, axis=0)
                    elif method == 'std':
                        dii_value = np.std(all_pixels, axis=0)
                    else:
                        raise ValueError(f"Unknown aggregation method: {method}")
                        
                    dii_results[cluster_id] = dii_value
    
    return dii_results


def process_superpixel_pipeline(image: np.ndarray,
                              segmentation_method: str = 'slic',
                              n_segments: int = 1000,
                              pca_components: int = 5,
                              eps: float = 0.5,
                              min_samples: int = 5) -> dict:
    """
    Complete superpixel processing pipeline.

    Parameters:
    -----------
    image : np.ndarray
        Input image array
    segmentation_method : str, default='slic'
        Segmentation method ('slic', 'felzenszwalb', 'quickshift')
    n_segments : int, default=1000
        Number of segments for SLIC
    pca_components : int, default=5
        Number of PCA components
    eps : float, default=0.5
        DBSCAN epsilon parameter
    min_samples : int, default=5
        DBSCAN minimum samples parameter

    Returns:
    --------
    dict
        Results containing segments, patches, features, clusters, and DII
    """
    # Segmentation
    if segmentation_method == 'slic':
        segments = slic_segmentation(image, n_segments=n_segments)
    elif segmentation_method == 'felzenszwalb':
        segments = felzenszwalb_segmentation(image)
    elif segmentation_method == 'quickshift':
        segments = quickshift_segmentation(image)
    else:
        raise ValueError(f"Unknown segmentation method: {segmentation_method}")
    
    # Extract patches
    patches = extract_patches(image, segments)
    
    # PCA transformation
    features = pca_segments(patches, n_components=pca_components)
    
    # Clustering
    if features.size > 0:
        scaled_features = scale_features(features)
        clusters = cluster_segments(scaled_features, eps=eps, min_samples=min_samples)
    else:
        clusters = np.array([])
    
    # DII extraction
    dii_results = extract_dii(image, clusters, segments, patches) if clusters.size > 0 else {}
    
    return {
        'segments': segments,
        'patches': patches,
        'features': features,
        'clusters': clusters,
        'dii': dii_results
    }
