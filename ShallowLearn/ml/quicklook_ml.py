"""
QuickLook Filtering System for Unified Satellite API
Integrates thumbnail/PVI processing with configurable dimensionality reduction
"""

import os
import requests
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod

import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN, KMeans
from sklearn.mixture import GaussianMixture

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


@dataclass
class QuickLookConfig:
    """Configuration for QuickLook processing"""
    # Dimensionality reduction
    reduction_method: str = "pca"  # "pca", "tsne", "umap", "svd"
    n_components: Union[int, float] = 0.95  # Components or variance explained
    
    # Clustering
    clustering_method: str = "dbscan"  # "dbscan", "kmeans", "gmm"
    clustering_params: Dict[str, Any] = None
    
    # Image processing
    target_size: Tuple[int, int] = (343, 343)  # Native Sentinel-2 size
    normalize: bool = True
    
    # Thumbnail handling
    download_thumbnails: bool = True
    cache_dir: Optional[str] = None
    
    def __post_init__(self):
        if self.clustering_params is None:
            if self.clustering_method == "dbscan":
                self.clustering_params = {"eps": 50, "min_samples": 5}
            elif self.clustering_method == "kmeans":
                self.clustering_params = {"n_clusters": 4}
            elif self.clustering_method == "gmm":
                self.clustering_params = {"n_components": 4}


class DimensionalityReducer(ABC):
    """Abstract base for dimensionality reduction methods"""
    
    @abstractmethod
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        pass


class PCAReducer(DimensionalityReducer):
    def __init__(self, n_components: Union[int, float] = 0.95):
        self.pca = PCA(n_components=n_components)
        self.n_components = n_components
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        return self.pca.fit_transform(data)
    
    def get_name(self) -> str:
        return f"PCA(n_components={self.n_components})"


class TSNEReducer(DimensionalityReducer):
    def __init__(self, n_components: int = 2, **kwargs):
        self.n_components = n_components
        self.kwargs = kwargs
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        # t-SNE is computationally expensive, limit components to reasonable range
        n_components = min(self.n_components, 3)
        
        # Adjust perplexity based on sample size
        n_samples = data.shape[0]
        max_perplexity = (n_samples - 1) // 3  # Rule of thumb: perplexity < n_samples/3
        perplexity = min(self.kwargs.get('perplexity', 30), max_perplexity, 50)
        perplexity = max(perplexity, 5)  # Minimum perplexity
        
        # Update kwargs with adjusted perplexity
        adjusted_kwargs = self.kwargs.copy()
        adjusted_kwargs['perplexity'] = perplexity
        
        tsne = TSNE(n_components=n_components, **adjusted_kwargs)
        return tsne.fit_transform(data)
    
    def get_name(self) -> str:
        return f"t-SNE(n_components={self.n_components})"


class UMAPReducer(DimensionalityReducer):
    def __init__(self, n_components: int = 2, **kwargs):
        if not UMAP_AVAILABLE:
            raise ImportError("UMAP not available. Install with: pip install umap-learn")
        self.n_components = n_components
        self.kwargs = kwargs
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        reducer = umap.UMAP(n_components=self.n_components, **self.kwargs)
        return reducer.fit_transform(data)
    
    def get_name(self) -> str:
        return f"UMAP(n_components={self.n_components})"


class SVDReducer(DimensionalityReducer):
    def __init__(self, n_components: Union[int, float] = 0.95):
        from sklearn.decomposition import TruncatedSVD
        if isinstance(n_components, float):
            # For SVD, we need to determine components differently
            self.n_components = min(50, int(n_components * 100))  # Heuristic
        else:
            self.n_components = n_components
        self.svd = TruncatedSVD(n_components=self.n_components)
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        return self.svd.fit_transform(data)
    
    def get_name(self) -> str:
        return f"SVD(n_components={self.n_components})"


class ThumbnailLoader:
    """Handles thumbnail/PVI loading for different satellite types"""
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir or tempfile.mkdtemp()
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def load_thumbnail(self, product, target_size: Tuple[int, int] = (343, 343)) -> Optional[np.ndarray]:
        """Load thumbnail for a satellite product"""
        try:
            if product.thumbnail_url:
                return self._load_from_url(product.thumbnail_url, target_size)
            elif product.satellite == "sentinel2":
                return self._load_sentinel2_pvi(product, target_size)
            elif product.satellite == "landsat":
                return self._load_landsat_thumbnail(product, target_size)
            else:
                print(f"No thumbnail method for {product.satellite}")
                return None
        except Exception as e:
            print(f"Failed to load thumbnail for {product.product_id}: {e}")
            return None
    
    def _load_from_url(self, url: str, target_size: Tuple[int, int]) -> np.ndarray:
        """Load thumbnail from URL"""
        # Create cache filename from URL
        cache_file = os.path.join(self.cache_dir, f"{hash(url)}.jpg")
        
        if not os.path.exists(cache_file):
            response = requests.get(url)
            response.raise_for_status()
            with open(cache_file, 'wb') as f:
                f.write(response.content)
        
        with Image.open(cache_file) as img:
            img = img.convert('RGB')
            img = img.resize(target_size)
            return np.array(img)
    
    def _load_sentinel2_pvi(self, product, target_size: Tuple[int, int]) -> np.ndarray:
        """Load Sentinel-2 PVI from product metadata or zip file"""
        # This would integrate with the existing PVI_Dataloader logic
        # For now, create a placeholder that would work with actual PVI files
        if hasattr(product, 'metadata') and product.metadata:
            # Try to extract PVI URL from metadata
            props = product.metadata.get('properties', {})
            thumbnail = props.get('thumbnail')
            if thumbnail:
                return self._load_from_url(thumbnail, target_size)
        
        # Fallback: generate placeholder thumbnail
        return self._generate_placeholder_thumbnail(product, target_size)
    
    def _load_landsat_thumbnail(self, product, target_size: Tuple[int, int]) -> np.ndarray:
        """Load Landsat thumbnail - would integrate with Landsat thumbnail generation"""
        # Landsat thumbnails would need to be generated from the downloaded data
        # For now, create placeholder
        return self._generate_placeholder_thumbnail(product, target_size)
    
    def _generate_placeholder_thumbnail(self, product, target_size: Tuple[int, int]) -> np.ndarray:
        """Generate a placeholder thumbnail based on product metadata"""
        # Create a simple colored thumbnail based on cloud cover and date
        cloud_cover = getattr(product, 'cloud_cover', 50) / 100.0
        
        # Create gradient based on cloud cover
        thumbnail = np.zeros((*target_size, 3), dtype=np.uint8)
        
        # Blue channel increases with cloud cover
        thumbnail[:, :, 2] = int(255 * cloud_cover)
        # Green channel decreases with cloud cover  
        thumbnail[:, :, 1] = int(255 * (1 - cloud_cover))
        # Red stays constant
        thumbnail[:, :, 0] = 128
        
        return thumbnail


class QuickLookFilter:
    """Main QuickLook filtering system for satellite products"""
    
    def __init__(self, config: Optional[QuickLookConfig] = None):
        self.config = config or QuickLookConfig()
        self.thumbnail_loader = ThumbnailLoader(self.config.cache_dir)
        self.reducer = self._create_reducer()
        
        # State
        self.products = []
        self.thumbnails = []
        self.transformed_data = None
        self.labels = None
        self.class_dict = {}
    
    def _create_reducer(self) -> DimensionalityReducer:
        """Create dimensionality reducer based on config"""
        method = self.config.reduction_method.lower()
        n_components = self.config.n_components
        
        if method == "pca":
            return PCAReducer(n_components)
        elif method == "tsne":
            return TSNEReducer(n_components if isinstance(n_components, int) else 2)
        elif method == "umap":
            return UMAPReducer(n_components if isinstance(n_components, int) else 2)
        elif method == "svd":
            return SVDReducer(n_components)
        else:
            raise ValueError(f"Unknown reduction method: {method}")
    
    def process_products(self, products: List) -> Dict[str, List]:
        """Process satellite products through QuickLook pipeline
        
        Returns:
            Dictionary with cluster labels as keys and filtered product lists as values
        """
        print(f"Processing {len(products)} products with {self.reducer.get_name()}...")
        
        # Load thumbnails
        self.products = products
        self.thumbnails = []
        valid_indices = []
        
        for i, product in enumerate(products):
            thumbnail = self.thumbnail_loader.load_thumbnail(product, self.config.target_size)
            if thumbnail is not None:
                self.thumbnails.append(thumbnail)
                valid_indices.append(i)
        
        if not self.thumbnails:
            raise ValueError("No valid thumbnails loaded")
        
        print(f"Loaded {len(self.thumbnails)} valid thumbnails")
        
        # Keep only products with valid thumbnails
        self.products = [products[i] for i in valid_indices]
        
        # Prepare data for dimensionality reduction
        thumbnail_data = np.array(self.thumbnails)
        if self.config.normalize:
            thumbnail_data = thumbnail_data.astype(np.float32) / 255.0
        
        # Flatten for dimensionality reduction
        flattened_data = thumbnail_data.reshape(len(self.thumbnails), -1)
        
        # Apply dimensionality reduction
        print(f"Applying {self.reducer.get_name()}...")
        self.transformed_data = self.reducer.fit_transform(flattened_data)
        
        # Apply clustering
        print(f"Clustering with {self.config.clustering_method}...")
        self.labels = self._apply_clustering(self.transformed_data)
        
        # Generate class dictionary
        self._generate_class_dict()
        
        # Return filtered products by cluster
        return self._group_products_by_cluster()
    
    def _apply_clustering(self, data: np.ndarray) -> np.ndarray:
        """Apply clustering algorithm"""
        method = self.config.clustering_method.lower()
        params = self.config.clustering_params
        
        if method == "dbscan":
            clusterer = DBSCAN(**params)
        elif method == "kmeans":
            clusterer = KMeans(**params)
        elif method == "gmm":
            clusterer = GaussianMixture(**params)
            # GMM returns probabilities, we take argmax
            labels = clusterer.fit_predict(data)
            return labels
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        return clusterer.fit_predict(data)
    
    def _generate_class_dict(self):
        """Generate class dictionary for visualization - semantic labels only for PCA"""
        unique_labels = np.unique(self.labels)
        
        # Check if we're using PCA method for semantic labeling
        is_pca = hasattr(self.reducer, 'get_name') and 'PCA' in self.reducer.get_name()
        
        if is_pca:
            # Calculate mean brightness for each cluster to determine cloud/clear classification
            cluster_means = {}
            for label in unique_labels:
                if label == -1:  # DBSCAN noise
                    continue
                mask = self.labels == label
                cluster_thumbnails = np.array(self.thumbnails)[mask]
                cluster_means[label] = np.mean(cluster_thumbnails)
            
            # Sort clusters by brightness (darker = clearer, brighter = cloudier)
            if cluster_means:
                sorted_clusters = sorted(cluster_means.items(), key=lambda x: x[1])
                
                # Assign semantic labels for PCA only
                self.class_dict = {-1: ("#808080", "Noise")}  # Gray for noise
                
                colors = ["#2ca02c", "#ff7f0e", "#d62728", "#1f77b4"]  # Green, orange, red, blue
                labels_semantic = ["Clear Sky", "Partially Cloudy", "Cloudy", "Very Cloudy"]
                
                for i, (cluster_id, _) in enumerate(sorted_clusters):
                    color_idx = i % len(colors)
                    label_idx = min(i, len(labels_semantic) - 1)
                    self.class_dict[cluster_id] = (colors[color_idx], labels_semantic[label_idx])
            else:
                # Fallback for no valid clusters in PCA
                self._generate_generic_labels(unique_labels)
        else:
            # For non-PCA methods, use generic cluster labels
            self._generate_generic_labels(unique_labels)
    
    def _generate_generic_labels(self, unique_labels):
        """Generate generic cluster labels for non-PCA methods"""
        colors = ["#2ca02c", "#ff7f0e", "#d62728", "#1f77b4", "#9467bd", "#8c564b", "#e377c2"]
        self.class_dict = {}
        
        for i, label in enumerate(unique_labels):
            if label == -1:  # DBSCAN noise
                self.class_dict[label] = ("#808080", "Noise")
            else:
                color_idx = i % len(colors)
                self.class_dict[label] = (colors[color_idx], f"Cluster {label}")
    
    def _group_products_by_cluster(self) -> Dict[str, List]:
        """Group products by their cluster labels"""
        clusters = {}
        
        for product, label in zip(self.products, self.labels):
            label_name = self.class_dict.get(label, (None, f"Cluster_{label}"))[1]
            
            if label_name not in clusters:
                clusters[label_name] = []
            
            clusters[label_name].append(product)
        
        return clusters
    
    def get_clear_sky_products(self) -> List:
        """Get products classified as clear sky"""
        clear_key = None
        for label, (_, name) in self.class_dict.items():
            if "clear" in name.lower():
                clear_key = label
                break
        
        if clear_key is not None:
            mask = self.labels == clear_key
            return [self.products[i] for i in range(len(self.products)) if mask[i]]
        else:
            print("No clear sky cluster identified")
            return []
    
    def get_clustering_summary(self) -> Dict[str, int]:
        """Get summary of clustering results"""
        summary = {}
        for label in np.unique(self.labels):
            count = np.sum(self.labels == label)
            label_name = self.class_dict.get(label, (None, f"Cluster_{label}"))[1]
            summary[label_name] = count
        
        return summary