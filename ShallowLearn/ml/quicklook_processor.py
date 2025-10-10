"""
Unified QuickLook processing for satellite imagery with extensible dimensionality reduction.
Compatible with both Sentinel-2 and Landsat data classes.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Union

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans

# Dimensionality reduction imports
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture

from ShallowLearn.utilities.util import clip_image

from ..core.array_utils import LCE_multi

# Local imports
from ..io.satellite_data import Sentinel2Image
from ..load.landsatdata_class import LandSatImage


class SatelliteImage(Protocol):
    """Protocol defining the common interface for satellite image classes."""

    image: np.ndarray
    tags: Dict[str, Any]
    meta: Dict[str, Any]
    present_bands: set
    band_status: Dict[str, bool]
    path: Path


class DimensionalityReducer(ABC):
    """Abstract base class for dimensionality reduction methods."""

    @abstractmethod
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit reducer and transform data."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get human-readable name of the method."""
        pass


class PCAReducer(DimensionalityReducer):
    """PCA dimensionality reduction."""

    def __init__(self, n_components: Union[int, float] = 0.95):
        self.n_components = n_components
        self.model = PCA(n_components=n_components)

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        return self.model.fit_transform(data)

    def get_name(self) -> str:
        if isinstance(self.n_components, float):
            return f"PCA ({self.n_components:.0%} variance)"
        else:
            return f"PCA ({self.n_components} components)"


class TSNEReducer(DimensionalityReducer):
    """t-SNE dimensionality reduction."""

    def __init__(
        self, n_components: int = 2, perplexity: float = 30.0, random_state: int = 42
    ):
        self.n_components = n_components
        self.perplexity = perplexity
        self.random_state = random_state

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        model = TSNE(
            n_components=self.n_components,
            perplexity=self.perplexity,
            random_state=self.random_state,
        )
        return model.fit_transform(data)

    def get_name(self) -> str:
        return f"t-SNE (perplexity={self.perplexity})"


class ClusteringMethod(ABC):
    """Abstract base class for clustering methods."""

    @abstractmethod
    def fit_predict(self, data: np.ndarray) -> np.ndarray:
        """Fit clustering model and predict labels."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get human-readable name of the method."""
        pass


class DBSCANClustering(ClusteringMethod):
    """DBSCAN clustering method."""

    def __init__(self, eps: float = 30, min_samples: int = 5):
        self.eps = eps
        self.min_samples = min_samples

    def fit_predict(self, data: np.ndarray) -> np.ndarray:
        model = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        return model.fit_predict(data)

    def get_name(self) -> str:
        return f"DBSCAN (eps={self.eps}, min_samples={self.min_samples})"


class KMeansClustering(ClusteringMethod):
    """K-Means clustering method."""

    def __init__(self, n_clusters: int = 4, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state

    def fit_predict(self, data: np.ndarray) -> np.ndarray:
        model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        return model.fit_predict(data)

    def get_name(self) -> str:
        return f"K-Means (k={self.n_clusters})"


class GMMClustering(ClusteringMethod):
    """Gaussian Mixture Model clustering method."""

    def __init__(self, n_components: int = 4, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state

    def fit_predict(self, data: np.ndarray) -> np.ndarray:
        model = GaussianMixture(
            n_components=self.n_components, random_state=self.random_state
        )
        return model.fit_predict(data)

    def get_name(self) -> str:
        return f"GMM (components={self.n_components})"


class SatelliteImageProcessor:
    """Processes satellite images for QuickLook analysis."""

    def __init__(
        self,
        target_bands: Optional[List[str]] = None,
        apply_stretch: bool = True,
        clip_percent: float = 2.0,
        normalize: bool = True,
    ):
        """
        Initialize image processor.

        Args:
            target_bands: Specific bands to use (None for RGB bands)
            apply_stretch: Whether to apply LCE stretch
            clip_percent: Percentage for pixel value clipping
            normalize: Whether to normalize pixel values to [0,1]
        """
        self.target_bands = target_bands
        self.apply_stretch = apply_stretch
        self.clip_percent = clip_percent
        self.normalize = normalize

    def get_default_bands(self, satellite_type: str) -> List[str]:
        """Get default RGB bands for satellite type."""
        if satellite_type == "sentinel-2":
            return ["B04", "B03", "B02"]  # RGB for Sentinel-2
        elif satellite_type == "landsat":
            return ["B4", "B3", "B2"]  # RGB for Landsat
        else:
            # Fallback - assume first 3 bands are RGB-like
            return []

    def detect_satellite_type(self, image: SatelliteImage) -> str:
        """Detect satellite type from image metadata."""
        # Check for Sentinel-2 indicators
        if hasattr(image, "tags"):
            processing_level = image.tags.get("PROCESSING_LEVEL", "")
            if any(
                level in processing_level
                for level in ["Level-1C", "Level-2A", "L1C", "L2A"]
            ):
                return "sentinel-2"

        # Check for Landsat indicators in path
        path_str = str(image.path).lower()
        if any(sat in path_str for sat in ["lc08", "lc09", "le07", "lt05"]):
            return "landsat"

        # Check band names
        sentinel_bands = {
            "B01",
            "B02",
            "B03",
            "B04",
            "B05",
            "B06",
            "B07",
            "B08",
            "B8A",
            "B09",
            "B11",
            "B12",
        }
        landsat_bands = {
            "B1",
            "B2",
            "B3",
            "B4",
            "B5",
            "B6",
            "B7",
            "B8",
            "B9",
            "B10",
            "B11",
        }

        if image.present_bands & sentinel_bands:
            return "sentinel-2"
        elif image.present_bands & landsat_bands:
            return "landsat"

        return "unknown"

    def process_image(self, image: SatelliteImage) -> Optional[np.ndarray]:
        """
        Process a single satellite image for QuickLook analysis.

        Args:
            image: Satellite image object (Sentinel2Image or LandSatImage)

        Returns:
            Processed image array or None if processing failed
        """
        try:
            satellite_type = self.detect_satellite_type(image)

            # Determine bands to use
            if self.target_bands is None:
                bands_to_use = self.get_default_bands(satellite_type)
            else:
                bands_to_use = self.target_bands

            # Handle case where no bands specified and we couldn't detect defaults
            if not bands_to_use:
                # Use first available bands up to 3
                available_bands = sorted(list(image.present_bands))
                bands_to_use = available_bands[:3]

            # Extract band indices
            if satellite_type == "sentinel-2":
                band_order = image.band_order
            elif satellite_type == "landsat":
                band_order = image.band_order
            else:
                # Create a simple band order based on available bands
                band_order = {
                    band: i for i, band in enumerate(sorted(image.present_bands))
                }

            # Get band indices for requested bands
            band_indices = []
            for band in bands_to_use:
                if band in image.present_bands and band in band_order:
                    band_indices.append(band_order[band])

            if not band_indices:
                print(f"No valid bands found for {image.path}")
                return None

            # Extract data for selected bands
            processed_image = image.image[:, :, band_indices]

            # Apply processing steps
            if self.apply_stretch and processed_image.dtype in [np.uint16, np.int16]:
                # Remove offset for Sentinel-2 (typically 1000)
                if satellite_type == "sentinel-2":
                    processed_image = processed_image.astype(np.float32)
                    processed_image = np.maximum(processed_image - 1000, 0)

                # Apply LCE stretch
                processed_image = LCE_multi(processed_image)
                if self.normalize:
                    processed_image = processed_image / 255.0

            if self.clip_percent > 0:
                processed_image = clip_image(
                    processed_image, clip_percent=self.clip_percent
                )

            return processed_image

        except Exception as e:
            print(f"Error processing {image.path}: {e}")
            return None


class QuickLookProcessor:
    """
    Unified QuickLook processor for satellite imagery analysis.
    Supports both Sentinel-2 and Landsat with extensible dimensionality reduction.
    """

    def __init__(
        self,
        reducer: Optional[DimensionalityReducer] = None,
        clustering: Optional[ClusteringMethod] = None,
        image_processor: Optional[SatelliteImageProcessor] = None,
    ):
        """
        Initialize QuickLook processor.

        Args:
            reducer: Dimensionality reduction method (default: PCA)
            clustering: Clustering method (default: DBSCAN)
            image_processor: Image processing pipeline
        """
        self.reducer = reducer or PCAReducer(n_components=0.95)
        self.clustering = clustering or DBSCANClustering(eps=50, min_samples=5)
        self.image_processor = image_processor or SatelliteImageProcessor()

        # Results storage
        self.images: List[SatelliteImage] = []
        self.processed_images: List[np.ndarray] = []
        self.transformed_data: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None
        self.metadata_df: Optional[pd.DataFrame] = None

    def process_images(
        self, images: List[Union[SatelliteImage, str]], create_metadata: bool = True
    ) -> "QuickLookProcessor":
        """
        Process a list of satellite images.

        Args:
            images: List of satellite image objects or file paths
            create_metadata: Whether to create metadata DataFrame

        Returns:
            Self for method chaining
        """
        print(f"Processing {len(images)} images...")

        self.images = []
        self.processed_images = []

        for img in images:
            try:
                # Load image if path provided
                if isinstance(img, (str, Path)):
                    img_path = Path(img)
                    # Detect and load appropriate image type
                    if any(
                        pattern in str(img_path).lower()
                        for pattern in ["s2a_", "s2b_", "msil1c", "msil2a", ".safe"]
                    ):
                        satellite_img = Sentinel2Image(str(img_path))
                    else:
                        satellite_img = LandSatImage(str(img_path))
                else:
                    satellite_img = img

                # Process image
                processed = self.image_processor.process_image(satellite_img)

                if processed is not None:
                    self.images.append(satellite_img)
                    self.processed_images.append(processed)

            except Exception as e:
                print(f"Failed to process image {img}: {e}")

        print(f"Successfully processed {len(self.processed_images)} images")

        if create_metadata:
            self._create_metadata_dataframe()

        return self

    def reduce_dimensions(self) -> "QuickLookProcessor":
        """Apply dimensionality reduction to processed images."""
        if not self.processed_images:
            raise ValueError(
                "No processed images available. Call process_images first."
            )

        print(
            f"Applying {self.reducer.get_name()} to {len(self.processed_images)} images..."
        )

        # Flatten images for dimensionality reduction
        flattened = np.array([img.flatten() for img in self.processed_images])

        # Apply dimensionality reduction
        self.transformed_data = self.reducer.fit_transform(flattened)

        print(
            f"Reduced from {flattened.shape[1]} to {self.transformed_data.shape[1]} dimensions"
        )

        return self

    def cluster_images(self) -> "QuickLookProcessor":
        """Apply clustering to transformed data."""
        if self.transformed_data is None:
            raise ValueError(
                "No transformed data available. Call reduce_dimensions first."
            )

        print(f"Applying {self.clustering.get_name()} clustering...")

        # Apply clustering
        self.labels = self.clustering.fit_predict(self.transformed_data)

        # Update metadata with labels
        if self.metadata_df is not None:
            self.metadata_df["cluster_label"] = self.labels

        unique_labels = np.unique(self.labels)
        print(f"Found {len(unique_labels)} clusters: {unique_labels}")

        return self

    def _create_metadata_dataframe(self) -> None:
        """Create metadata DataFrame from processed images."""
        metadata_list = []

        for img in self.images:
            metadata = {
                "file_path": str(img.path),
                "satellite_type": self.image_processor.detect_satellite_type(img),
                "present_bands": len(img.present_bands),
                "image_shape": img.image.shape,
            }

            # Add satellite-specific metadata
            if hasattr(img, "tags") and img.tags:
                metadata.update(
                    {
                        "processing_level": img.tags.get("PROCESSING_LEVEL", "Unknown"),
                        "aot_retrieval_method": img.tags.get(
                            "AOT_RETRIEVAL_METHOD", "N/A"
                        ),
                    }
                )

            # Add MTL metadata for Landsat
            if hasattr(img, "mtl_tags") and img.mtl_tags:
                metadata.update(
                    {
                        "landsat_scene_id": img.mtl_tags.get(
                            "LANDSAT_SCENE_ID", "Unknown"
                        ),
                        "cloud_cover": img.mtl_tags.get("CLOUD_COVER", -1),
                    }
                )

            metadata_list.append(metadata)

        self.metadata_df = pd.DataFrame(metadata_list)

    def get_results(self) -> Dict[str, Any]:
        """Get all processing results."""
        return {
            "images": self.images,
            "processed_images": self.processed_images,
            "transformed_data": self.transformed_data,
            "labels": self.labels,
            "metadata_df": self.metadata_df,
            "reducer_name": self.reducer.get_name(),
            "clustering_name": self.clustering.get_name(),
        }

    def run_complete_analysis(
        self, images: List[Union[SatelliteImage, str]]
    ) -> "QuickLookProcessor":
        """
        Run complete QuickLook analysis pipeline.

        Args:
            images: List of satellite images or file paths

        Returns:
            Self with all analysis complete
        """
        return self.process_images(images).reduce_dimensions().cluster_images()


# Convenience function for quick analysis
def quick_analysis(
    image_paths: List[str],
    method: str = "pca",
    clustering: str = "dbscan",
    target_bands: Optional[List[str]] = None,
) -> QuickLookProcessor:
    """
    Convenience function for quick satellite image analysis.

    Args:
        image_paths: List of paths to satellite images
        method: Dimensionality reduction method ('pca', 'tsne')
        clustering: Clustering method ('dbscan', 'kmeans', 'gmm')
        target_bands: Specific bands to analyze

    Returns:
        Completed QuickLookProcessor instance
    """
    # Set up dimensionality reduction
    if method.lower() == "pca":
        reducer = PCAReducer()
    elif method.lower() == "tsne":
        reducer = TSNEReducer()
    else:
        reducer = PCAReducer()

    # Set up clustering
    if clustering.lower() == "dbscan":
        cluster_method = DBSCANClustering()
    elif clustering.lower() == "kmeans":
        cluster_method = KMeansClustering()
    elif clustering.lower() == "gmm":
        cluster_method = GMMClustering()
    else:
        cluster_method = DBSCANClustering()

    # Set up image processor
    image_processor = SatelliteImageProcessor(target_bands=target_bands)

    # Create and run processor
    processor = QuickLookProcessor(
        reducer=reducer, clustering=cluster_method, image_processor=image_processor
    )

    return processor.run_complete_analysis(image_paths)

