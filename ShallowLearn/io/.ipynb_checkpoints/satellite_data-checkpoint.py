"""
Satellite data classes for Landsat and Sentinel-2 with consistent interfaces.
Handles missing bands, metadata preservation, and VRT generation.
"""

import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import rasterio as rio
from osgeo import gdal
from PIL import Image


class SatelliteImage(ABC):
    """Abstract base class for satellite images with consistent interface."""

    def __init__(self, file_path: str):
        self.path = Path(file_path)
        self.meta = {}
        self.tags = {}
        self.present_bands = set()
        self.band_status = {}
        self.image = None

        self._load_image()

    @property
    @abstractmethod
    def band_order(self) -> Dict[str, int]:
        """Define the canonical band order for this satellite type."""
        pass

    @abstractmethod
    def _load_image(self):
        """Load image data with satellite-specific logic."""
        pass

    def __repr__(self):
        band_list = [
            f"{b} {'✓' if self.band_status.get(b, False) else '✗'}"
            for b in sorted(self.band_order, key=lambda x: self.band_order[x])
        ]
        missing_count = sum(not v for v in self.band_status.values())
        return (
            f"<{self.__class__.__name__}: {self.path}\n"
            f"  Bands: {band_list}\n"
            f"  Shape: {self.image.shape if self.image is not None else 'Not loaded'}\n"
            f"  Missing: {missing_count} bands>"
        )

    def get_band_data(self, band_name: str) -> Optional[np.ndarray]:
        """Get data for a specific band."""
        if band_name not in self.band_order:
            raise ValueError(f"Unknown band: {band_name}")

        band_index = self.band_order[band_name]
        if self.image is not None and band_index < self.image.shape[2]:
            return self.image[:, :, band_index]
        return None

    def has_band(self, band_name: str) -> bool:
        """Check if band is present (not a placeholder)."""
        return self.band_status.get(band_name, False)

    def get_metadata(self) -> Dict:
        """Get image metadata."""
        return self.meta if hasattr(self, 'meta') and self.meta else {}

    def get_bounds(self):
        """Get image bounds."""
        if hasattr(self, 'meta') and self.meta and 'transform' in self.meta:
            # Calculate bounds from transform and dimensions
            transform = self.meta['transform']
            width = self.meta['width']
            height = self.meta['height']
            
            # Calculate corner coordinates
            left = transform.c
            top = transform.f
            right = transform.c + width * transform.a
            bottom = transform.f + height * transform.e
            
            # Return bounds in a format similar to rasterio.coords.BoundingBox
            from rasterio.coords import BoundingBox
            return BoundingBox(left=left, bottom=bottom, right=right, top=top)
        return None

    def get_rgb_bands(self) -> Tuple[str, str, str]:
        """Get the typical RGB band combination for this satellite."""
        # Default implementation - should be overridden by subclasses
        return ("B4", "B3", "B2")  # Red, Green, Blue for most satellites


class LandsatImage(SatelliteImage):
    """Landsat image with strict band ordering and missing band handling."""

    @property
    def band_order(self) -> Dict[str, int]:
        """Canonical Landsat band order with index mapping."""
        return {
            "B1": 0,  # Coastal/Aerosol
            "B2": 1,  # Blue
            "B3": 2,  # Green
            "B4": 3,  # Red
            "B5": 4,  # NIR
            "B6": 5,  # SWIR1 (Landsat 8/9) or Thermal (Landsat 7)
            "B7": 6,  # SWIR2
            "B8": 7,  # Panchromatic (Landsat 8/9)
            "B9": 8,  # Cirrus (Landsat 8/9)
            "B10": 9,  # Thermal 1 (Landsat 8/9)
            "B11": 10,  # Thermal 2 (Landsat 8/9)
            "SAA": 11,  # Solar Azimuth Angle
            "SZA": 12,  # Solar Zenith Angle
            "VAA": 13,  # View Azimuth Angle
            "VZA": 14,  # View Zenith Angle
            "PIXEL": 15,  # Pixel QA
            "RADSAT": 16,  # Radiometric Saturation
        }

    def _load_image(self):
        """Load Landsat image data from VRT file."""
        self.mtl_tags = {}

        with rio.open(self.path) as src:
            # Store metadata
            self.meta = src.meta.copy()
            self.tags = src.tags()
            self.mtl_tags = src.tags(ns="MTL")

            # Read bands individually to handle mixed dtypes in Landsat VRTs
            # VRTs can contain spectral bands + metadata bands with different dtypes
            band_data = {}
            for i in range(src.count):
                band_desc = (
                    src.descriptions[i] if src.descriptions[i] else f"Band_{i + 1}"
                )
                if band_desc in self.band_order:
                    if band_desc in band_data.keys():
                        continue
                    # Read individual band to handle mixed dtypes
                    try:
                        band_array = src.read(i + 1)
                        band_data[band_desc] = band_array
                        self.present_bands.add(band_desc)
                    except Exception as e:
                        print(f"Warning: Could not read band {band_desc}: {e}")
                        continue

            # Create ordered array with placeholders for missing bands
            self._create_ordered_array(band_data)

    def _create_ordered_array(self, band_data: Dict[str, np.ndarray]):
        """Create ordered array with NaN placeholders for missing bands."""
        ordered_bands = []
        self.band_status = {}

        for band_name in sorted(self.band_order, key=lambda x: self.band_order[x]):
            if band_name in band_data:
                ordered_bands.append(band_data[band_name])
                self.band_status[band_name] = True
            else:
                # Create zero placeholder with same shape/dtype as existing bands
                if band_data:
                    first_band = next(iter(band_data.values()))
                    placeholder = np.zeros_like(first_band)
                else:
                    placeholder = np.zeros(
                        (self.meta["height"], self.meta["width"]), dtype="uint16"
                    )
                ordered_bands.append(placeholder)
                self.band_status[band_name] = False

        self.image = np.stack(ordered_bands, axis=0)
        # (height, width, channels)
        self.image = np.transpose(self.image, (1, 2, 0))

    def get_rgb_bands(self) -> Tuple[str, str, str]:
        """Get RGB band combination for Landsat."""
        return ("B4", "B3", "B2")  # Red, Green, Blue

    def get_spectral_bands(self) -> List[str]:
        """Get list of spectral bands (excluding QA and angle bands)."""
        spectral_bands = ["B1", "B2", "B3", "B4", "B5", "B6", "B7"]
        if self.has_band("B8"):  # Landsat 8/9 has panchromatic
            spectral_bands.append("B8")
        if self.has_band("B9"):  # Landsat 8/9 has cirrus
            spectral_bands.append("B9")
        return spectral_bands


class Sentinel2Image(SatelliteImage):
    """Sentinel-2 image with band ordering and missing band handling."""

    @property
    def band_order(self) -> Dict[str, int]:
        """Canonical Sentinel-2 band order with index mapping."""
        return {
            "B01": 0,  # Coastal aerosol (60m)
            "B02": 1,  # Blue (10m)
            "B03": 2,  # Green (10m)
            "B04": 3,  # Red (10m)
            "B05": 4,  # Red Edge 1 (20m)
            "B06": 5,  # Red Edge 2 (20m)
            "B07": 6,  # Red Edge 3 (20m)
            "B08": 7,  # NIR (10m)
            "B8A": 8,  # NIR narrow (20m)
            "B09": 9,  # Water vapour (60m)
            "B10": 10,  # Cirrus (60m)
            "B11": 11,  # SWIR 1 (20m)
            "B12": 12,  # SWIR 2 (20m)
        }

    def _load_image(self):
        """Load Sentinel-2 image data using subdatasets approach like original LoadSentinel2L1C."""
        import zipfile
        import os
        
        # Handle different input types
        if str(self.path).endswith(".vrt"):
            # Handle VRT files created by the VRT builder
            self.is_vrt = True
            with rio.open(self.path) as src:
                # Store metadata
                self.meta = src.meta.copy()
                self.tags = src.tags()
                
                # Read bands individually to handle any mixed dtypes
                band_data = {}
                for i in range(src.count):
                    band_desc = (
                        src.descriptions[i] if src.descriptions[i] else f"Band_{i + 1}"
                    )
                    # Handle the naming convention (B2 vs B02)
                    if band_desc.startswith('B') and len(band_desc) == 2:
                        band_desc = f"B0{band_desc[1]}"
                    
                    if band_desc in self.band_order:
                        try:
                            band_array = src.read(i + 1)
                            band_data[band_desc] = band_array
                            self.present_bands.add(band_desc)
                        except Exception as e:
                            print(f"Warning: Could not read band {band_desc}: {e}")
                            continue

                # Create ordered array with placeholders for missing bands
                self._create_ordered_array(band_data)
                return
                
        elif str(self.path).endswith(".zip"):
            self.is_zip = True
            with zipfile.ZipFile(self.path, 'r') as zip_ref:
                files = [
                    f for f in zip_ref.namelist()
                    if "MTD_MSIL1C.xml" in f or "MTD_MSIL2A.xml" in f
                ]
            if len(files) != 1:
                raise Exception("Multiple or no MTD files found in ZIP.")
            
            zip_path = f"/vsizip/{self.path}"
            metadata_file = os.path.join(zip_path, files[0])
        else:
            # Handle .SAFE directories or direct XML files
            self.is_zip = False
            if str(self.path).endswith(".xml"):
                metadata_file = str(self.path)
            else:
                # Assume it's a .SAFE directory - use simple file finding for now
                # This avoids import issues with missing modules
                from pathlib import Path
                safe_path = Path(self.path)
                mtd_files = list(safe_path.rglob("MTD_MSIL1C*.xml"))
                if not mtd_files:
                    mtd_files = list(safe_path.rglob("MTD_MSIL2A*.xml"))
                if len(mtd_files) != 1:
                    raise Exception(f"Found {len(mtd_files)} MTD files, expected 1")
                metadata_file = str(mtd_files[0])

        # Load subdatasets using the original approach
        with rio.open(metadata_file) as dataset:
            subdatasets = dataset.subdatasets

        if not subdatasets:
            raise ValueError("No subdatasets found in the Sentinel-2 file")

        # Get metadata from the first subdataset
        with rio.open(subdatasets[0]) as ds:
            self.tags = ds.tags()
            self.meta = ds.meta.copy()
            
        # Use 10m resolution bands as default (like original implementation)
        resolution_10m = [s for s in subdatasets if "10m" in s]
        
        if resolution_10m:
            # Load the 10m resolution data
            with rio.open(resolution_10m[0]) as src:
                # Read all bands at once - this maintains original resolution
                data = src.read()  # Shape: (bands, height, width)
                
                # Map bands based on descriptions
                band_data = {}
                for i in range(src.count):
                    band_desc = (
                        src.descriptions[i].split(",")[0] if src.descriptions[i] else f"Band_{i + 1}"
                    )
                    # Handle the naming convention (B2 vs B02)
                    if band_desc.startswith('B') and len(band_desc) == 2:
                        band_desc = f"B0{band_desc[1]}"
                    
                    if band_desc in self.band_order:
                        band_data[band_desc] = data[i]
                        self.present_bands.add(band_desc)

                # Create ordered array with placeholders for missing bands
                self._create_ordered_array(band_data)
        else:
            # Fallback: use the first available subdataset
            with rio.open(subdatasets[0]) as src:
                data = src.read()
                self.image = np.transpose(data, (1, 2, 0))  # (height, width, bands)

    def _create_ordered_array(self, band_data: Dict[str, np.ndarray]):
        """Create ordered array with NaN placeholders for missing bands."""
        ordered_bands = []
        self.band_status = {}

        for band_name in sorted(self.band_order, key=lambda x: self.band_order[x]):
            if band_name in band_data:
                ordered_bands.append(band_data[band_name])
                self.band_status[band_name] = True
            else:
                # Create zero placeholder (avoid NaN casting issues with integer dtypes)
                if band_data:
                    first_band = next(iter(band_data.values()))
                    placeholder = np.zeros_like(first_band)
                else:
                    placeholder = np.zeros(
                        (self.meta["height"], self.meta["width"]), dtype="uint16"
                    )
                ordered_bands.append(placeholder)
                self.band_status[band_name] = False

        self.image = np.stack(ordered_bands, axis=0)
        # (height, width, channels)
        self.image = np.transpose(self.image, (1, 2, 0))

    def get_rgb_bands(self) -> Tuple[str, str, str]:
        """Get RGB band combination for Sentinel-2."""
        return ("B04", "B03", "B02")  # Red, Green, Blue

    def get_spectral_bands(self) -> List[str]:
        """Get list of all spectral bands."""
        return list(self.band_order.keys())

    def get_resolution_groups(self) -> Dict[str, List[str]]:
        """Get bands grouped by native resolution."""
        return {
            "10m": ["B02", "B03", "B04", "B08"],
            "20m": ["B05", "B06", "B07", "B8A", "B11", "B12"],
            "60m": ["B01", "B09", "B10"],
        }


class SatelliteImageCollection(ABC):
    """Abstract base class for collections of satellite images."""

    def __init__(self, directory: str):
        self.directory = Path(directory)
        self.image_files = self._get_sorted_image_files()
        self.images = [self._create_image(f) for f in self.image_files]

    @abstractmethod
    def _get_sorted_image_files(self) -> List[Path]:
        """Get sorted list of image files."""
        pass

    @abstractmethod
    def _create_image(self, file_path: Path) -> SatelliteImage:
        """Create appropriate satellite image instance."""
        pass

    def __iter__(self):
        return iter(self.images)

    def __getitem__(self, index):
        return self.images[index]

    def __len__(self):
        return len(self.images)

    def __repr__(self):
        return f"<{self.__class__.__name__} count={len(self)}>"

    def common_bands(self) -> Set[str]:
        """Get bands present in ALL images."""
        if not self.images:
            return set()
        return set.intersection(*[img.present_bands for img in self.images])

    def get_common_bands_array(self) -> np.ndarray:
        """Get array of common bands across all images with consistent spatial dimensions."""
        common_bands = self.common_bands()
        if not common_bands or not self.images:
            return np.array([])

        # Get canonical band indices for common bands
        first_image = self.images[0]
        band_indices = sorted([first_image.band_order[b] for b in common_bands])

        # Find maximum spatial dimensions
        max_height = max(img.image.shape[0] for img in self.images)
        max_width = max(img.image.shape[1] for img in self.images)

        # Resize and stack images
        resized_images = []
        for img in self.images:
            resized_bands = []
            for channel in range(img.image.shape[2]):
                band = img.image[:, :, channel]

                # Convert to PIL Image and resize if needed
                if band.shape != (max_height, max_width):
                    pil_band = Image.fromarray(band)
                    resized_band = pil_band.resize(
                        (max_width, max_height), Image.BILINEAR
                    )
                    resized_bands.append(np.array(resized_band))
                else:
                    resized_bands.append(band)

            # Stack resized bands and select common channels
            resized_img = np.stack(resized_bands, axis=2)
            resized_images.append(resized_img[:, :, band_indices])

        return np.stack(resized_images, axis=0)


class LandsatImageCollection(SatelliteImageCollection):
    """Managed collection of Landsat images with date sorting and strict band order."""

    def _get_sorted_image_files(self) -> List[Path]:
        """Get Landsat VRT files sorted by date."""

        def extract_date(filename):
            """Extract date from Landsat filename."""
            parts = filename.stem.split("_")
            if len(parts) > 3 and re.match(r"\d{8}", parts[3]):
                return parts[3]
            return ""

        # Look for VRT files, exclude Landsat 7 if needed
        files = [f for f in self.directory.glob("*.vrt") if "LE07" not in f.name]
        return sorted(files, key=extract_date)

    def _create_image(self, file_path: Path) -> LandsatImage:
        """Create LandsatImage instance."""
        return LandsatImage(file_path)


class Sentinel2ImageCollection(SatelliteImageCollection):
    """Managed collection of Sentinel-2 images with date sorting."""

    def _get_sorted_image_files(self) -> List[Path]:
        """Get Sentinel-2 VRT files sorted by date."""

        def extract_date(filename):
            """Extract date from Sentinel-2 filename."""
            # S2A_MSIL2A_20210101T000000_... pattern
            parts = filename.stem.split("_")
            for part in parts:
                if len(part) >= 8 and part.startswith("20") and part[8:9] == "T":
                    return part[:8]  # Extract YYYYMMDD
            return ""

        files = list(self.directory.glob("*.vrt"))
        return sorted(files, key=extract_date)

    def _create_image(self, file_path: Path) -> Sentinel2Image:
        """Create Sentinel2Image instance."""
        return Sentinel2Image(file_path)


def create_satellite_image(file_path: str) -> SatelliteImage:
    """
    Factory function to create appropriate satellite image based on file path.

    Parameters:
    -----------
    file_path : str
        Path to the satellite image file

    Returns:
    --------
    SatelliteImage
        Appropriate satellite image instance
    """
    path = Path(file_path)
    filename = path.name.upper()

    # Detect satellite type from filename patterns
    if any(sat in filename for sat in ["LC08", "LC09", "LE07", "LT05", "LT04"]):
        return LandsatImage(file_path)
    elif any(sat in filename for sat in ["S2A", "S2B"]):
        return Sentinel2Image(file_path)
    else:
        # Try to detect from file content or default to Landsat
        # This could be enhanced with more sophisticated detection
        return LandsatImage(file_path)


def create_satellite_collection(
    directory: str, satellite_type: Optional[str] = None
) -> SatelliteImageCollection:
    """
    Factory function to create appropriate satellite image collection.

    Parameters:
    -----------
    directory : str
        Directory containing satellite images
    satellite_type : str, optional
        Force specific satellite type ('landsat' or 'sentinel2')

    Returns:
    --------
    SatelliteImageCollection
        Appropriate satellite image collection
    """
    if satellite_type:
        if satellite_type.lower() == "landsat":
            return LandsatImageCollection(directory)
        elif satellite_type.lower() in ["sentinel2", "sentinel-2", "s2"]:
            return Sentinel2ImageCollection(directory)
        else:
            raise ValueError(f"Unknown satellite type: {satellite_type}")

    # Auto-detect based on files in directory
    path = Path(directory)
    files = list(path.glob("*.vrt"))

    if not files:
        raise ValueError(f"No VRT files found in {directory}")

    # Check first file for satellite type
    first_file = files[0].name.upper()
    if any(sat in first_file for sat in ["LC08", "LC09", "LE07", "LT05", "LT04"]):
        return LandsatImageCollection(directory)
    elif any(sat in first_file for sat in ["S2A", "S2B"]):
        return Sentinel2ImageCollection(directory)
    else:
        # Default to Landsat if uncertain
        return LandsatImageCollection(directory)


class GeoTIFFImage:
    """
    Generic GeoTIFF loader with backwards compatibility to LoadGeoTIFF.
    
    Supports various GeoTIFF types including:
    - Planetscope individual band files
    - GBR benthic classification data
    - Generic single/multi-band GeoTIFF files
    """
    
    def __init__(self, file_path: str):
        """
        Initialize GeoTIFF loader.
        
        Parameters:
        -----------
        file_path : str
            Path to the GeoTIFF file
        """
        self.data_source = file_path  # Maintain compatibility with LoadGeoTIFF
        self.path = Path(file_path)
        self.metadata = None
        self.bounds = None
        self.image = None
        
        if not self.path.exists():
            raise FileNotFoundError(f"GeoTIFF file not found: {file_path}")
            
    def load(self) -> np.ndarray:
        """
        Load GeoTIFF data with backwards compatibility.
        
        Returns:
        --------
        np.ndarray
            Image data with shape (bands, height, width)
        """
        try:
            with rio.open(self.data_source) as src:
                # Read all bands
                data = src.read()
                
                # Handle nodata values similar to original LoadGeoTIFF
                no_data = src.nodatavals
                if no_data and any(nd is not None for nd in no_data):
                    # Create mask for nodata values but don't apply it yet
                    # (maintaining compatibility with original behavior)
                    pass
                    
                self.image = data
                return data
                
        except Exception as e:
            raise RuntimeError(f"Failed to load GeoTIFF {self.data_source}: {e}")
    
    def get_metadata(self) -> dict:
        """
        Get rasterio metadata for the GeoTIFF file.
        
        Returns:
        --------
        dict
            Rasterio metadata dictionary
        """
        try:
            with rio.open(self.data_source) as src:
                self.metadata = src.meta.copy()
            return self.metadata
        except Exception as e:
            raise RuntimeError(f"Failed to get metadata for {self.data_source}: {e}")
    
    def get_bounds(self) -> rio.coords.BoundingBox:
        """
        Get spatial bounds of the GeoTIFF file.
        
        Returns:
        --------
        rasterio.coords.BoundingBox
            Bounding box (left, bottom, right, top)
        """
        try:
            with rio.open(self.data_source) as src:
                self.bounds = src.bounds
            return self.bounds
        except Exception as e:
            raise RuntimeError(f"Failed to get bounds for {self.data_source}: {e}")
    
    def get_crs(self):
        """
        Get coordinate reference system.
        
        Returns:
        --------
        rasterio.crs.CRS
            Coordinate reference system
        """
        try:
            with rio.open(self.data_source) as src:
                return src.crs
        except Exception as e:
            raise RuntimeError(f"Failed to get CRS for {self.data_source}: {e}")
    
    def get_transform(self):
        """
        Get affine transform.
        
        Returns:
        --------
        rasterio.Affine
            Affine transformation
        """
        try:
            with rio.open(self.data_source) as src:
                return src.transform
        except Exception as e:
            raise RuntimeError(f"Failed to get transform for {self.data_source}: {e}")
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Get image shape without loading full data."""
        if self.image is not None:
            return self.image.shape
        try:
            with rio.open(self.data_source) as src:
                return (src.count, src.height, src.width)
        except Exception as e:
            raise RuntimeError(f"Failed to get shape for {self.data_source}: {e}")
    
    @property
    def dtype(self):
        """Get image data type without loading full data."""
        try:
            with rio.open(self.data_source) as src:
                return src.dtypes[0]  # Assume all bands have same dtype
        except Exception as e:
            raise RuntimeError(f"Failed to get dtype for {self.data_source}: {e}")
    
    def __repr__(self) -> str:
        shape_str = f"{self.shape}" if hasattr(self, 'shape') else "Unknown"
        return f"<GeoTIFFImage: {self.path.name}, Shape: {shape_str}>"


class GeoTIFFCollection:
    """
    Collection manager for multiple GeoTIFF files.
    
    Useful for handling datasets like Planetscope with multiple single-band files
    or collections of classification/analysis results.
    """
    
    def __init__(self, directory: str, pattern: str = "*.tif"):
        """
        Initialize collection from directory.
        
        Parameters:
        -----------
        directory : str
            Directory containing GeoTIFF files
        pattern : str, default "*.tif"
            Glob pattern to match files
        """
        self.directory = Path(directory)
        self.pattern = pattern
        self.files = []
        self.images = []
        
        if not self.directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
            
        self._discover_files()
    
    def _discover_files(self):
        """Discover and sort GeoTIFF files in directory."""
        self.files = sorted(list(self.directory.glob(self.pattern)))
        
        if not self.files:
            raise ValueError(f"No files matching pattern '{self.pattern}' found in {self.directory}")
    
    def load_all(self) -> List[GeoTIFFImage]:
        """
        Load all GeoTIFF files in the collection.
        
        Returns:
        --------
        List[GeoTIFFImage]
            List of loaded GeoTIFF images
        """
        self.images = []
        for file_path in self.files:
            try:
                img = GeoTIFFImage(str(file_path))
                img.load()
                self.images.append(img)
            except Exception as e:
                print(f"Warning: Failed to load {file_path}: {e}")
                continue
        
        return self.images
    
    def get_file_list(self) -> List[Path]:
        """Get list of discovered files."""
        return self.files.copy()
    
    def stack_images(self) -> np.ndarray:
        """
        Stack all images into a single array.
        
        Returns:
        --------
        np.ndarray
            Stacked images with shape (n_images, bands, height, width)
        """
        if not self.images:
            self.load_all()
        
        if not self.images:
            raise ValueError("No images successfully loaded")
        
        # Assume all images have compatible shapes
        stacked = np.stack([img.image for img in self.images], axis=0)
        return stacked
    
    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, index: int) -> GeoTIFFImage:
        """Get image by index, loading if necessary."""
        if index >= len(self.files):
            raise IndexError(f"Index {index} out of range for {len(self.files)} files")
        
        if index >= len(self.images):
            # Load missing images up to requested index
            for i in range(len(self.images), index + 1):
                img = GeoTIFFImage(str(self.files[i]))
                img.load()
                self.images.append(img)
        
        return self.images[index]
    
    def __repr__(self) -> str:
        return f"<GeoTIFFCollection: {len(self.files)} files in {self.directory.name}>"


# Backwards compatibility alias
LoadGeoTIFF = GeoTIFFImage
