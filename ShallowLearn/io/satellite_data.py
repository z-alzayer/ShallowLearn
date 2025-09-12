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

    def __init__(self, file_path: str, load_all_bands: bool = False, target_resolution: str = "10m", 
                 clip_geometry=None, buffer_meters: float = 0):
        """
        Initialize Sentinel-2 image.
        
        Parameters:
        -----------
        file_path : str
            Path to Sentinel-2 file (.SAFE directory, .zip file, or MTD XML file)
        load_all_bands : bool
            If True, loads all 13 bands by resampling from different resolution subdatasets.
            If False, loads only the native resolution bands (default: 4 bands at 10m)
        target_resolution : str
            Target resolution when load_all_bands=True ("10m", "20m", "60m")
        clip_geometry : shapely geometry or GeoDataFrame, optional
            Geometry to clip to during loading for efficiency
        buffer_meters : float
            Buffer distance in meters to add around clip_geometry
        """
        self.load_all_bands = load_all_bands
        self.target_resolution = target_resolution
        self.clip_geometry = clip_geometry
        self.buffer_meters = buffer_meters
        super().__init__(file_path)

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
            
        # Load bands based on configuration
        if self.load_all_bands:
            # Load all bands from multiple resolution subdatasets
            self._load_all_resolution_bands(subdatasets, self.target_resolution)
        else:
            # Use 10m resolution bands as default (original behavior)
            resolution_10m = [s for s in subdatasets if "10m" in s]
            
            if resolution_10m:
                # Calculate clipping window if geometry provided
                clip_window = None
                if self.clip_geometry is not None:
                    clip_window = self._calculate_clip_window(resolution_10m[0])
                    if clip_window is None:
                        print("Warning: Invalid clip window, loading full 10m image")
                
                # Load the 10m resolution data
                with rio.open(resolution_10m[0]) as src:
                    try:
                        # Read all bands (with clipping if specified)
                        if clip_window:
                            data = src.read(window=clip_window)  # Shape: (bands, height, width)
                            # Update metadata for clipped region
                            from rasterio.windows import transform as window_transform
                            self.meta['transform'] = window_transform(clip_window, src.meta['transform'])
                            self.meta['width'] = int(clip_window.width)
                            self.meta['height'] = int(clip_window.height)
                        else:
                            data = src.read()  # Shape: (bands, height, width)
                    except Exception as e:
                        print(f"Error reading 10m data: {e}")
                        if clip_window:
                            print(f"Window: {clip_window.width} x {clip_window.height}")
                        # Fallback to full image
                        data = src.read()
                    
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
    
    def _load_all_resolution_bands(self, subdatasets: List[str], target_resolution: str = "10m"):
        """
        Load all bands from multiple resolution subdatasets and resample to target resolution.
        Applies clipping during loading if clip_geometry is specified for efficiency.
        
        Parameters:
        -----------
        subdatasets : List[str]
            List of subdataset URIs
        target_resolution : str
            Target resolution to resample all bands to ("10m", "20m", "60m")
        """
        from rasterio.warp import reproject, Resampling
        from rasterio.enums import Resampling as ResamplingEnum
        from rasterio.windows import from_bounds
        
        # Find subdatasets by resolution
        resolution_subdatasets = {}
        for subdataset in subdatasets:
            if ":10m:" in subdataset:
                resolution_subdatasets["10m"] = subdataset
            elif ":20m:" in subdataset:
                resolution_subdatasets["20m"] = subdataset
            elif ":60m:" in subdataset:
                resolution_subdatasets["60m"] = subdataset
        
        if not resolution_subdatasets:
            raise ValueError("No resolution subdatasets found")
        
        # Get target resolution parameters
        target_subdataset = resolution_subdatasets.get(target_resolution)
        if not target_subdataset:
            # Fallback to 10m if target not available
            target_resolution = "10m"
            target_subdataset = resolution_subdatasets.get(target_resolution)
        
        if not target_subdataset:
            raise ValueError("No suitable target resolution found")
        
        # Calculate clipping window if geometry provided
        clip_window = None
        if self.clip_geometry is not None:
            clip_window = self._calculate_clip_window(target_subdataset)
            if clip_window is None:
                print("Warning: Invalid clip window, loading full image")
        
        # Get target grid parameters (from clipped region if applicable)
        with rio.open(target_subdataset) as target_ds:
            if clip_window:
                target_width = int(clip_window.width)
                target_height = int(clip_window.height)
                from rasterio.windows import transform as window_transform
                target_transform = window_transform(clip_window, target_ds.transform)
            else:
                target_width = target_ds.width
                target_height = target_ds.height
                target_transform = target_ds.transform
            target_crs = target_ds.crs
        
        # Load and resample all bands
        band_data = {}
        for resolution, subdataset in resolution_subdatasets.items():
            with rio.open(subdataset) as src:
                # Calculate appropriate window for this resolution
                if clip_window and resolution != target_resolution:
                    # Scale window to match resolution ratio
                    scale_factor = self._get_resolution_scale_factor(target_resolution, resolution)
                    res_window = self._scale_window(clip_window, scale_factor)
                else:
                    res_window = clip_window
                
                for i in range(src.count):
                    # Parse band name from description
                    band_desc = src.descriptions[i].split(",")[0] if src.descriptions[i] else f"Band_{i + 1}"
                    # Handle naming convention (B2 vs B02)
                    if band_desc.startswith('B') and len(band_desc) == 2:
                        band_desc = f"B0{band_desc[1]}"
                    
                    if band_desc in self.band_order:
                        try:
                            # Read the band (with clipping if window specified)
                            if res_window:
                                band_array = src.read(i + 1, window=res_window)
                            else:
                                band_array = src.read(i + 1)
                        except Exception as e:
                            print(f"Error reading band {band_desc} from {resolution}: {e}")
                            if res_window:
                                print(f"Window: {res_window.width} x {res_window.height}")
                            continue
                        
                        if resolution == target_resolution:
                            # No resampling needed
                            band_data[band_desc] = band_array
                        else:
                            # Resample to target resolution
                            resampled_array = np.empty((target_height, target_width), dtype=band_array.dtype)
                            
                            # Get source transform for this window
                            if res_window:
                                from rasterio.windows import transform as window_transform
                                src_transform = window_transform(res_window, src.transform)
                            else:
                                src_transform = src.transform
                            
                            reproject(
                                band_array,
                                resampled_array,
                                src_transform=src_transform,
                                src_crs=src.crs,
                                dst_transform=target_transform,
                                dst_crs=target_crs,
                                resampling=ResamplingEnum.bilinear
                            )
                            band_data[band_desc] = resampled_array
                        
                        self.present_bands.add(band_desc)
        
        # Create ordered array with placeholders for missing bands
        self._create_ordered_array(band_data)
    
    def _calculate_clip_window(self, reference_subdataset: str):
        """Calculate clipping window from geometry for the reference subdataset."""
        import geopandas as gpd
        from rasterio.windows import from_bounds
        
        # Handle different geometry types
        if hasattr(self.clip_geometry, 'geometry'):
            # It's a GeoDataFrame
            gdf = self.clip_geometry
        else:
            # It's a geometry - create a GeoDataFrame
            gdf = gpd.GeoDataFrame([1], geometry=[self.clip_geometry], crs="EPSG:4326")
        
        # Get CRS from reference subdataset
        with rio.open(reference_subdataset) as ref_ds:
            target_crs = ref_ds.crs
            
            # Reproject geometry to match image CRS if needed
            if gdf.crs != target_crs:
                gdf = gdf.to_crs(target_crs)
            
            # Apply buffer if specified
            if self.buffer_meters > 0:
                gdf_buffered = gdf.copy()
                gdf_buffered.geometry = gdf.geometry.buffer(self.buffer_meters)
                bounds = gdf_buffered.total_bounds
            else:
                bounds = gdf.total_bounds
            
            # Calculate window from bounds
            window = from_bounds(
                bounds[0], bounds[1], bounds[2], bounds[3],  # left, bottom, right, top
                transform=ref_ds.transform
            )
            
            # Round and clip to dataset bounds
            from rasterio.windows import Window
            window = window.round_lengths().round_offsets()
            dataset_window = Window(0, 0, ref_ds.width, ref_ds.height)
            window = window.intersection(dataset_window)
            
            # Ensure window has valid dimensions
            if window.width <= 0 or window.height <= 0:
                print(f"Warning: Invalid window dimensions: {window.width} x {window.height}")
                print(f"Bounds: {bounds}")
                print(f"Dataset size: {ref_ds.width} x {ref_ds.height}")
                print(f"Transform: {ref_ds.transform}")
                return None
            
            print(f"Calculated clip window: {window.width} x {window.height} at ({window.col_off}, {window.row_off})")
            return window
    
    def _get_resolution_scale_factor(self, target_res: str, source_res: str) -> float:
        """Get scale factor between resolutions."""
        resolution_values = {"10m": 10, "20m": 20, "60m": 60}
        return resolution_values[source_res] / resolution_values[target_res]
    
    def _scale_window(self, window, scale_factor: float):
        """Scale a window by the given factor."""
        from rasterio.windows import Window
        if window is None:
            return None
            
        scaled = Window(
            col_off=window.col_off / scale_factor,  # Inverse scaling for higher resolution
            row_off=window.row_off / scale_factor,
            width=window.width / scale_factor,
            height=window.height / scale_factor
        ).round_lengths().round_offsets()
        
        # Ensure valid dimensions
        if scaled.width <= 0 or scaled.height <= 0:
            print(f"Warning: Invalid scaled window: {scaled.width} x {scaled.height}")
            return None
            
        return scaled
    
    def clip_to_bounds(self, bounds, buffer_pixels: int = 0):
        """
        Clip image data to specified bounds.
        
        Parameters:
        -----------
        bounds : tuple or BoundingBox
            Bounds to clip to (left, bottom, right, top) or rasterio BoundingBox
        buffer_pixels : int
            Number of pixels to add as buffer around the clipped area
            
        Returns:
        --------
        Sentinel2Image
            New Sentinel2Image instance with clipped data
        """
        if self.image is None or not hasattr(self, 'meta'):
            raise ValueError("Image and metadata must be loaded before clipping")
        
        from rasterio.coords import BoundingBox
        from rasterio.windows import from_bounds
        from copy import deepcopy
        
        # Ensure bounds is a BoundingBox
        if not isinstance(bounds, BoundingBox):
            bounds = BoundingBox(*bounds)
        
        # Calculate window from bounds
        window = from_bounds(
            bounds.left, bounds.bottom, bounds.right, bounds.top,
            transform=self.meta['transform']
        )
        
        # Apply buffer if specified
        if buffer_pixels > 0:
            window = window.expand(buffer_pixels)
        
        # Round window to integer pixels
        window = window.round_lengths().round_offsets()
        
        # Clip the window to image boundaries  
        from rasterio.windows import Window
        image_window = Window(0, 0, self.meta['width'], self.meta['height'])
        window = window.intersection(image_window)
        
        if window.width <= 0 or window.height <= 0:
            raise ValueError("Clipping bounds do not intersect with image")
        
        # Extract the clipped data
        row_slice = slice(int(window.row_off), int(window.row_off + window.height))
        col_slice = slice(int(window.col_off), int(window.col_off + window.width))
        
        clipped_image = self.image[row_slice, col_slice, :]
        
        # Create new instance with clipped data
        clipped_s2 = self.__class__.__new__(self.__class__)
        clipped_s2.path = self.path
        clipped_s2.present_bands = self.present_bands.copy()
        clipped_s2.band_status = self.band_status.copy()
        clipped_s2.tags = self.tags.copy()
        clipped_s2.image = clipped_image
        
        # Update metadata
        clipped_s2.meta = deepcopy(self.meta)
        clipped_s2.meta['width'] = int(window.width)
        clipped_s2.meta['height'] = int(window.height)
        
        # Update transform
        from rasterio.windows import transform as window_transform
        clipped_s2.meta['transform'] = window_transform(window, self.meta['transform'])
        
        return clipped_s2
    
    def clip_to_geometry(self, geometry, buffer_meters: float = 0):
        """
        Clip image data to a geometry (e.g., from a GeoDataFrame).
        
        Parameters:
        -----------
        geometry : shapely geometry or GeoDataFrame
            Geometry to clip to
        buffer_meters : float
            Buffer distance in meters to add around the geometry
            
        Returns:
        --------
        Sentinel2Image
            New Sentinel2Image instance with clipped data
        """
        import geopandas as gpd
        from shapely.geometry import box
        
        # Handle GeoDataFrame input
        if hasattr(geometry, 'geometry'):
            # It's a GeoDataFrame
            gdf = geometry
        else:
            # It's a geometry - create a GeoDataFrame
            gdf = gpd.GeoDataFrame([1], geometry=[geometry], crs="EPSG:4326")
        
        # Reproject to image CRS if needed
        if gdf.crs != self.meta['crs']:
            gdf = gdf.to_crs(self.meta['crs'])
        
        # Apply buffer if specified
        if buffer_meters > 0:
            gdf_buffered = gdf.copy()
            gdf_buffered.geometry = gdf.geometry.buffer(buffer_meters)
            geometry_bounds = gdf_buffered.total_bounds
        else:
            geometry_bounds = gdf.total_bounds
        
        # Clip to bounding box first
        return self.clip_to_bounds(geometry_bounds)


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
