"""
GeoTIFF compilers for creating multi-band composites from satellite data.
Follows similar pattern to VRT builders but creates physical GeoTIFF files.
"""

import os
import tempfile
import shutil
import zipfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rasterio as rio
from rasterio.enums import Resampling
from rasterio import open as rio_open

from ShallowLearn.core.band_mapping import band_mapping
from .file_processing import (
    unzip_files, get_file_names_from_zip, delete_files_from_dir, 
    list_files_in_dir, filter_files_by_extension, check_values_in_filenames,
    order_by_band, list_files_in_dir_recur, order_band_names_noreg
)


class GeoTIFFCompiler(ABC):
    """Abstract base class for GeoTIFF compilers."""

    def __init__(self, output_dir: str, project_name: str = "Satellite Processing"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.project_name = project_name

    @abstractmethod
    def compile_geotiff(
        self, source_path: str, output_name: str, **kwargs
    ) -> str:
        """Compile multi-band GeoTIFF from satellite source."""
        pass

    @abstractmethod
    def _get_band_files(self, source_path: str) -> List[str]:
        """Get list of band files from source."""
        pass

    @abstractmethod
    def _get_reference_band(self, band_files: List[str], high_res_band: str = "B02") -> str:
        """Get reference band for metadata and dimensions."""
        pass


class ImageCompiler:
    """
    Helper class for compiling a set of images into one multi-band image.
    Modernized version of the original ImageCompiler.
    """

    def __init__(self, image_paths: List[str], reference_band_path: str, output_path: str):
        """
        Create a new ImageCompiler instance.
        
        Parameters:
        -----------
        image_paths : List[str]
            Paths of the images to compile
        reference_band_path : str
            The path of the band with the highest resolution
        output_path : str
            Path where the output should be saved
        """
        self.image_paths = image_paths
        self.reference_band_path = reference_band_path
        self.output_path = output_path
        
        self._get_common_metadata()

    def _get_common_metadata(self):
        """Fetch common metadata from the reference band."""
        with rio_open(self.reference_band_path) as src:
            self.crs = src.crs
            self.height = src.height
            self.width = src.width
            self.transform = src.transform
            self.count = src.count
            self.bands = len(self.image_paths)
            self.dtype = src.dtypes[0]
            self.bounds = src.bounds

    def compile_to_file(self) -> str:
        """
        Compile the images into a single stacked GeoTIFF file.
        
        Returns:
        --------
        str
            Path to the compiled GeoTIFF
        """
        # Create output profile
        profile = {
            'driver': 'GTiff',
            'width': self.width,
            'height': self.height,
            'count': self.bands,
            'crs': self.crs,
            'transform': self.transform,
            'dtype': self.dtype,
            'compress': 'lzw',  # Add compression
            'tiled': True,      # Use tiling for better performance
            'blockxsize': 512,
            'blockysize': 512
        }
        
        with rio_open(self.output_path, 'w', **profile) as dst:
            for i, band_path in enumerate(self.image_paths, 1):
                with rio_open(band_path) as src:
                    # Read the band data
                    if src.count > 1:
                        # Multi-band file - use first band
                        data = src.read(1)
                    else:
                        data = src.read(1)
                    
                    # Resample if dimensions don't match reference
                    if src.width != self.width or src.height != self.height:
                        # Use rasterio's reproject for proper resampling
                        from rasterio.warp import reproject
                        
                        resampled_data = np.empty((self.height, self.width), dtype=data.dtype)
                        reproject(
                            source=data,
                            destination=resampled_data,
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=self.transform,
                            dst_crs=self.crs,
                            resampling=Resampling.cubic
                        )
                        dst.write(resampled_data, i)
                    else:
                        dst.write(data, i)
        
        return self.output_path


class Sentinel2GeoTIFFCompiler(GeoTIFFCompiler):
    """GeoTIFF compiler for Sentinel-2 data."""

    def __init__(self, output_dir: str, project_name: str = "Sentinel-2 Processing", band_order: Dict = None):
        super().__init__(output_dir, project_name)
        self.band_order = band_order if band_order else band_mapping

    def compile_geotiff(
        self, 
        source_path: str, 
        output_name: str, 
        high_res_band: str = "B02",
        **kwargs
    ) -> str:
        """
        Compile Sentinel-2 data into a multi-band GeoTIFF.
        
        Parameters:
        -----------
        source_path : str
            Path to Sentinel-2 ZIP file or extracted directory
        output_name : str
            Name for the output GeoTIFF file
        high_res_band : str
            Reference band for spatial resolution (default: "B02")
            
        Returns:
        --------
        str
            Path to compiled GeoTIFF file
        """
        source_path = Path(source_path)
        output_path = self.output_dir / output_name
        
        if source_path.suffix.lower() == '.zip':
            return self._process_zip_file(str(source_path), str(output_path), high_res_band)
        elif source_path.is_dir():
            if any(f.suffix == '.SAFE' for f in source_path.iterdir()):
                return self._process_extracted_safe(str(source_path), str(output_path), high_res_band)
            else:
                return self._process_sen2cor_local(str(source_path), str(output_path), high_res_band)
        else:
            raise ValueError(f"Unsupported source path type: {source_path}")

    def _process_zip_file(self, zip_path: str, output_path: str, high_res_band: str) -> str:
        """
        Process Sentinel-2 ZIP file and generate a GeoTIFF.
        Uses L2A imagery that is already processed by ESA.
        """
        # Create temporary directory for extraction
        with tempfile.TemporaryDirectory() as temp_dir:
            # Filter and extract image files from zip
            image_files = filter_files_by_extension(get_file_names_from_zip(zip_path), ".jp2")
            image_files = check_values_in_filenames(image_files, self.band_order)
            ordered_image_files = order_by_band([f for f in image_files if "/IMG_DATA" in f])
            
            # Extract files to temporary directory
            unzip_files(zip_path, ordered_image_files, temp_dir)
            
            # Create full paths for extracted files
            extracted_files = [os.path.join(temp_dir, f) for f in ordered_image_files]
            
            # Get reference band for compilation
            reference_band = self._get_reference_band(extracted_files, high_res_band)
            
            # Compile the GeoTIFF
            compiler = ImageCompiler(extracted_files, reference_band, output_path)
            return compiler.compile_to_file()

    def _process_extracted_safe(self, safe_path: str, output_path: str, high_res_band: str) -> str:
        """
        Process extracted Sentinel-2 .SAFE directory.
        """
        # Find image files in the SAFE directory
        image_files = list_files_in_dir_recur(safe_path)
        image_files = check_values_in_filenames(image_files, self.band_order)
        ordered_image_files = order_band_names_noreg([f for f in image_files if "IMG_DATA" in f and f.endswith(".jp2")])
        
        # Get reference band
        reference_band = self._get_reference_band(ordered_image_files, high_res_band)
        
        # Compile the GeoTIFF
        compiler = ImageCompiler(ordered_image_files, reference_band, output_path)
        return compiler.compile_to_file()

    def _process_sen2cor_local(self, directory_path: str, output_path: str, high_res_band: str) -> str:
        """
        Process locally processed Sentinel-2 data (e.g., Sen2Cor output).
        """
        # Find image files
        image_files = list_files_in_dir_recur(directory_path)
        image_files = check_values_in_filenames(image_files, self.band_order)
        ordered_image_files = order_by_band([f for f in image_files if "IMG_DATA" in f])
        
        # Get reference band
        reference_band = self._get_reference_band(ordered_image_files, high_res_band)
        
        # Compile the GeoTIFF
        compiler = ImageCompiler(ordered_image_files, reference_band, output_path)
        return compiler.compile_to_file()

    def _get_band_files(self, source_path: str) -> List[str]:
        """Get list of band files from Sentinel-2 source."""
        source_path = Path(source_path)
        
        if source_path.suffix.lower() == '.zip':
            image_files = filter_files_by_extension(get_file_names_from_zip(str(source_path)), ".jp2")
            return [f for f in image_files if "/IMG_DATA" in f]
        elif source_path.is_dir():
            image_files = list_files_in_dir_recur(str(source_path))
            return [f for f in image_files if "IMG_DATA" in f and f.endswith((".jp2", ".tif", ".tiff"))]
        else:
            return []

    def _get_reference_band(self, band_files: List[str], high_res_band: str = "B02") -> str:
        """Get reference band path for spatial dimensions."""
        # Try to find the specified high resolution band
        for band_file in band_files:
            if high_res_band in os.path.basename(band_file):
                return band_file
        
        # Fallback to first available 10m band
        high_res_bands = ["B02", "B03", "B04", "B08"]
        for ref_band in high_res_bands:
            for band_file in band_files:
                if ref_band in os.path.basename(band_file):
                    return band_file
        
        # Final fallback to first file
        return band_files[0] if band_files else None


class LandsatGeoTIFFCompiler(GeoTIFFCompiler):
    """GeoTIFF compiler for Landsat data."""

    def compile_geotiff(
        self, 
        source_path: str, 
        output_name: str, 
        high_res_band: str = "B4",
        **kwargs
    ) -> str:
        """
        Compile Landsat data into a multi-band GeoTIFF.
        
        Parameters:
        -----------
        source_path : str
            Path to Landsat tar file or extracted directory
        output_name : str
            Name for the output GeoTIFF file
        high_res_band : str
            Reference band for spatial resolution (default: "B4")
            
        Returns:
        --------
        str
            Path to compiled GeoTIFF file
        """
        # Implementation would be similar to Sentinel-2 but for Landsat data
        # This is a placeholder for future implementation
        raise NotImplementedError("Landsat GeoTIFF compilation not yet implemented")

    def _get_band_files(self, source_path: str) -> List[str]:
        """Get list of band files from Landsat source."""
        # Placeholder implementation
        return []

    def _get_reference_band(self, band_files: List[str], high_res_band: str = "B4") -> str:
        """Get reference band path for spatial dimensions."""
        # Placeholder implementation
        return band_files[0] if band_files else None


def create_geotiff_compiler(
    satellite_type: str, 
    output_dir: str, 
    **kwargs
) -> GeoTIFFCompiler:
    """
    Factory function to create appropriate GeoTIFF compiler.

    Parameters:
    -----------
    satellite_type : str
        Type of satellite ("landsat" or "sentinel2")
    output_dir : str
        Output directory for GeoTIFF files
    **kwargs
        Additional arguments for compiler

    Returns:
    --------
    GeoTIFFCompiler
        Appropriate GeoTIFF compiler instance
    """
    if satellite_type.lower() in ["sentinel2", "sentinel-2", "s2"]:
        return Sentinel2GeoTIFFCompiler(output_dir, **kwargs)
    elif satellite_type.lower() == "landsat":
        return LandsatGeoTIFFCompiler(output_dir, **kwargs)
    else:
        raise ValueError(f"Unknown satellite type: {satellite_type}")


def batch_compile_geotiffs(
    source_list: List[str],
    output_dir: str,
    satellite_type: Optional[str] = None,
    **kwargs,
) -> List[str]:
    """
    Batch compile multiple satellite sources to GeoTIFF files.

    Parameters:
    -----------
    source_list : List[str]
        List of source file/directory paths
    output_dir : str
        Output directory for GeoTIFF files
    satellite_type : str, optional
        Force specific satellite type
    **kwargs
        Additional arguments for compiler

    Returns:
    --------
    List[str]
        List of created GeoTIFF file paths
    """
    if not source_list:
        print("No sources provided")
        return []

    # Auto-detect satellite type if not provided
    if satellite_type is None:
        first_source = Path(source_list[0]).name.upper()
        if any(sat in first_source for sat in ["S2A", "S2B", "MSI"]):
            satellite_type = "sentinel2"
        elif any(sat in first_source for sat in ["LC08", "LC09", "LE07", "LT05", "LT04"]):
            satellite_type = "landsat"
        else:
            # Default to Sentinel-2 if uncertain
            satellite_type = "sentinel2"

    # Create compiler
    compiler = create_geotiff_compiler(satellite_type, output_dir, **kwargs)

    # Process each source
    created_geotiffs = []
    for i, source_path in enumerate(source_list):
        try:
            source_name = Path(source_path).stem
            output_name = f"{source_name}_compiled.tiff"
            
            geotiff_path = compiler.compile_geotiff(source_path, output_name, **kwargs)
            created_geotiffs.append(geotiff_path)
            print(f"Created: {geotiff_path}")
        except Exception as e:
            print(f"Failed to process {source_path}: {e}")

    print(f"Successfully created {len(created_geotiffs)} GeoTIFF files")
    return created_geotiffs