"""
VRT builders for Landsat and Sentinel-2 data with metadata preservation.
Handles creation of Virtual Datasets from satellite archives.
"""

import math
import os
import tarfile
import zipfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import rasterio
from osgeo import gdal
from pyproj import CRS, Transformer

# Enable GDAL exceptions for better error handling
gdal.UseExceptions()


class VRTBuilder(ABC):
    """Abstract base class for VRT builders."""

    def __init__(self, output_dir: str, project_name: str = "Satellite Processing"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.project_name = project_name

    @abstractmethod
    def build_vrt(
        self, archive_path: str, bounds: Optional[gpd.GeoDataFrame] = None, **kwargs
    ) -> str:
        """Build VRT from satellite archive."""
        pass

    @abstractmethod
    def _get_band_files(self, archive_path: str) -> List[str]:
        """Get list of band files from archive."""
        pass

    @abstractmethod
    def _parse_metadata(self, archive_path: str) -> Dict[str, Any]:
        """Parse metadata from archive."""
        pass

    def expand_bounds(
        self,
        bounds: Tuple[float, float, float, float],
        n_pixels: int = 10,
        pixel_size: float = 30,
    ) -> Tuple[float, float, float, float]:
        """
        Expand bounds (in lon/lat) by n_pixels * pixel_size meters in all directions.

        Parameters:
        -----------
        bounds : Tuple[float, float, float, float]
            Bounds as (minx, miny, maxx, maxy) in WGS84
        n_pixels : int
            Number of pixels to expand
        pixel_size : float
            Pixel size in meters

        Returns:
        --------
        Tuple[float, float, float, float]
            Expanded bounds
        """
        minx, miny, maxx, maxy = bounds
        mean_lat = (miny + maxy) / 2
        expand_m = n_pixels * pixel_size

        # Convert meters to degrees
        meters_per_degree_lat = 111320
        meters_per_degree_lon = 111320 * math.cos(math.radians(mean_lat))
        expand_deg_lat = expand_m / meters_per_degree_lat
        expand_deg_lon = expand_m / meters_per_degree_lon

        return (
            minx - expand_deg_lon,
            miny - expand_deg_lat,
            maxx + expand_deg_lon,
            maxy + expand_deg_lat,
        )

    def transform_bounds(
        self, bounds: Tuple[float, float, float, float], src_crs: str, dst_crs: str
    ) -> Tuple[float, float, float, float]:
        """
        Transform bounds from src_crs to dst_crs.

        Parameters:
        -----------
        bounds : Tuple[float, float, float, float]
            Bounds as (minx, miny, maxx, maxy)
        src_crs : str
            Source CRS
        dst_crs : str
            Destination CRS

        Returns:
        --------
        Tuple[float, float, float, float]
            Transformed bounds as (ulx, uly, lrx, lry)
        """
        minx, miny, maxx, maxy = bounds
        transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
        ulx, uly = transformer.transform(minx, maxy)
        lrx, lry = transformer.transform(maxx, miny)
        return ulx, uly, lrx, lry


class LandsatVRTBuilder(VRTBuilder):
    """VRT builder for Landsat tar archives."""

    def build_vrt(
        self,
        archive_path: str,
        bounds: Optional[gpd.GeoDataFrame] = None,
        n_pixels: int = 10,
        pixel_size: float = 30,
        **kwargs,
    ) -> str:
        """
        Build VRT from Landsat tar archive.

        Parameters:
        -----------
        archive_path : str
            Path to Landsat tar archive
        bounds : gpd.GeoDataFrame, optional
            Geographic bounds for cropping
        n_pixels : int
            Number of pixels to expand bounds
        pixel_size : float
            Pixel size in meters for Landsat

        Returns:
        --------
        str
            Path to created VRT file
        """
        archive_path = Path(archive_path)

        # Get band files and metadata
        band_files = self._get_band_files(str(archive_path))
        if not band_files:
            raise RuntimeError(f"No .TIF files found in {archive_path}")

        metadata = self._parse_metadata(str(archive_path))

        # Get raster CRS from first band
        raster_crs = self._get_raster_crs(str(archive_path), band_files[0])

        # Build VRT name
        vrt_base = archive_path.stem
        vrt_path = self.output_dir / f"{vrt_base}.vrt"

        # Build /vsitar/ paths for all bands
        vsi_paths = [f"/vsitar/{archive_path}/{band_file}" for band_file in band_files]

        # Create initial VRT
        vrt_ds = gdal.BuildVRT(str(vrt_path), vsi_paths, separate=True)
        vrt_ds = None

        # Crop VRT if bounds provided
        if bounds is not None:
            cropped_vrt_path = self.output_dir / f"{vrt_base}_cropped.vrt"

            # Expand and transform bounds
            expanded_bounds = self.expand_bounds(
                bounds.total_bounds, n_pixels, pixel_size
            )
            ulx, uly, lrx, lry = self.transform_bounds(
                expanded_bounds, "EPSG:4326", raster_crs.to_string()
            )

            # Crop VRT
            gdal.Translate(
                str(cropped_vrt_path),
                str(vrt_path),
                projWin=[ulx, uly, lrx, lry],
                format="VRT",
            )

            final_vrt_path = cropped_vrt_path
        else:
            final_vrt_path = vrt_path

        # Add metadata to VRT
        self._add_metadata_to_vrt(
            str(final_vrt_path), metadata, band_files, str(archive_path)
        )

        print(f"Created: {final_vrt_path}")
        return str(final_vrt_path)

    def _get_band_files(self, archive_path: str) -> List[str]:
        """Get sorted list of .TIF band files from tar archive."""
        with tarfile.open(archive_path, "r") as tar:
            members = tar.getnames()

        band_files = sorted([name for name in members if name.endswith(".TIF")])
        return band_files

    def _parse_metadata(self, archive_path: str) -> Dict[str, Any]:
        """Parse MTL metadata from Landsat tar archive."""
        mtl_dict = {}
        stack = []

        with tarfile.open(archive_path, "r") as tar:
            mtl_member = next(
                (m for m in tar.getmembers() if m.name.endswith("MTL.txt")), None
            )
            if mtl_member is None:
                raise FileNotFoundError("MTL.txt not found in tar archive")

            mtl_file = tar.extractfile(mtl_member)
            for line_bytes in mtl_file:
                line = line_bytes.decode("utf-8").strip()
                if not line or line == "END":
                    continue

                if line.startswith("GROUP = "):
                    group = line.split("=", 1)[1].strip()
                    stack.append(group)
                elif line.startswith("END_GROUP"):
                    if stack:
                        stack.pop()
                elif "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip('"')
                    full_key = ".".join(stack + [key])
                    mtl_dict[full_key] = value

        return mtl_dict

    def _get_raster_crs(self, archive_path: str, band_file: str) -> CRS:
        """Get CRS from a raster band in tar archive."""
        vsi_path = f"/vsitar/{archive_path}/{band_file}"
        src_ds = gdal.Open(vsi_path)
        raster_crs_wkt = src_ds.GetProjection()
        src_ds = None
        return CRS.from_wkt(raster_crs_wkt)

    def _add_metadata_to_vrt(
        self,
        vrt_path: str,
        metadata: Dict[str, Any],
        band_files: List[str],
        source_path: str,
    ):
        """Add metadata to VRT file."""
        vrt_ds = gdal.Open(vrt_path, gdal.GA_Update)

        # Add MTL metadata
        vrt_ds.SetMetadata(metadata, "MTL")

        # Add custom metadata
        custom_metadata = {
            "project": self.project_name,
            "source_archive": source_path,
            "band_count": str(len(band_files)),
            "descriptions": ", ".join([self._extract_band_id(f) for f in band_files]),
        }
        vrt_ds.SetMetadata(custom_metadata)

        # Set band descriptions
        band_descriptions = [self._extract_band_id(f) for f in band_files]
        for i, description in enumerate(band_descriptions, 1):
            if i <= vrt_ds.RasterCount:
                band = vrt_ds.GetRasterBand(i)
                band.SetDescription(description)
                band.SetMetadataItem("DESCRIPTION", description)

        vrt_ds = None

    def _extract_band_id(self, filename: str) -> str:
        """Extract band ID from Landsat filename."""
        return filename.split(".")[0].split("_")[-1]


class Sentinel2VRTBuilder(VRTBuilder):
    """VRT builder for Sentinel-2 ZIP archives using MTD XML subdatasets."""

    def build_vrt(
        self,
        archive_path: str,
        bounds: Optional[gpd.GeoDataFrame] = None,
        n_pixels: int = 10,
        pixel_size: float = 10,
        target_resolution: str = "10m",
        **kwargs,
    ) -> str:
        """
        Build VRT from Sentinel-2 ZIP archive using JP2 files.
        All bands are resampled to target resolution for consistent numpy array handling.

        Parameters:
        -----------
        archive_path : str
            Path to Sentinel-2 ZIP archive
        bounds : gpd.GeoDataFrame, optional
            Geographic bounds for cropping
        n_pixels : int
            Number of pixels to expand bounds
        pixel_size : float
            Pixel size in meters for Sentinel-2
        target_resolution : str
            Target resolution to resample all bands to ("10m", "20m", "60m")

        Returns:
        --------
        str
            Path to created VRT file
        """
        archive_path = Path(archive_path)

        # Get JP2 band files directly
        band_files = self._get_band_files(str(archive_path))
        if not band_files:
            raise RuntimeError(f"No JP2 files found in {archive_path}")

        # Parse metadata
        metadata = self._parse_metadata(str(archive_path))

        # Get the highest resolution bands to use as reference grid
        reference_bands = self._filter_by_resolution(band_files, target_resolution)
        if not reference_bands:
            raise RuntimeError(f"No reference bands found for resolution {target_resolution}")

        # Get CRS and dimensions from reference (highest resolution) band
        raster_crs = self._get_raster_crs(str(archive_path), reference_bands[0])
        
        # Get target dimensions from the reference resolution
        with rasterio.open(f"/vsizip/{archive_path}/{reference_bands[0]}") as src:
            target_width = src.width
            target_height = src.height
            target_transform = src.transform
            target_crs = src.crs

        # Build VRT name
        vrt_base = archive_path.stem
        vrt_path = self.output_dir / f"{vrt_base}_{target_resolution}.vrt"

        # Build /vsizip/ paths for ALL bands (not just target resolution)
        vsi_paths = [f"/vsizip/{archive_path}/{band_file}" for band_file in band_files]

        # Create VRT with resampling ALL bands to target resolution grid
        vrt_options = gdal.BuildVRTOptions(
            resolution='user',
            xRes=abs(target_transform.a),  # Use actual pixel size from reference
            yRes=abs(target_transform.e),
            outputBounds=[target_transform.c, 
                         target_transform.f + target_height * target_transform.e,
                         target_transform.c + target_width * target_transform.a,
                         target_transform.f],
            targetAlignedPixels=True,
            separate=True,
            resampleAlg='bilinear'
        )

        vrt_ds = gdal.BuildVRT(str(vrt_path), vsi_paths, options=vrt_options)
        vrt_ds = None

        # Crop VRT if bounds provided
        if bounds is not None:
            cropped_vrt_path = (
                self.output_dir / f"{vrt_base}_cropped_{target_resolution}.vrt"
            )

            # Expand and transform bounds
            expanded_bounds = self.expand_bounds(
                bounds.total_bounds, n_pixels, pixel_size
            )
            ulx, uly, lrx, lry = self.transform_bounds(
                expanded_bounds, "EPSG:4326", str(raster_crs)
            )

            # Crop VRT
            gdal.Translate(
                str(cropped_vrt_path),
                str(vrt_path),
                projWin=[ulx, uly, lrx, lry],
                format="VRT",
            )

            final_vrt_path = cropped_vrt_path
        else:
            final_vrt_path = vrt_path

        # Add metadata to VRT
        self._add_metadata_to_vrt(
            str(final_vrt_path), metadata, band_files, str(archive_path)
        )

        print(f"Created: {final_vrt_path}")
        return str(final_vrt_path)


    def _get_band_files(self, archive_path: str) -> List[str]:
        """Get list of JP2 band files from ZIP archive."""
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            all_files = zip_ref.namelist()
        
        # Get all JP2 files in IMG_DATA, exclude TCI
        band_files = [f for f in all_files 
                     if "IMG_DATA" in f and f.endswith(".jp2") and "TCI" not in f]
        
        # Sort by band number for consistent ordering
        band_files.sort(key=self._extract_band_number)
        return band_files
    
    def _extract_band_number(self, filename: str) -> float:
        """Extract band number for sorting."""
        basename = os.path.basename(filename)
        parts = basename.split("_")
        
        for part in parts:
            if part.startswith("B") and len(part) >= 2:
                band_num_str = part[1:]
                if band_num_str.isdigit():
                    return float(band_num_str)
                elif band_num_str == "8A":  # Handle B8A case
                    return 8.5
        
        return 999.0  # Fallback
    
    def _filter_by_resolution(self, band_files: List[str], resolution: str) -> List[str]:
        """Filter band files by resolution."""
        # Sentinel-2 band resolution mapping
        band_resolutions = {
            "B01": "60m", "B02": "10m", "B03": "10m", "B04": "10m",
            "B05": "20m", "B06": "20m", "B07": "20m", "B08": "10m",
            "B8A": "20m", "B09": "60m", "B10": "60m", "B11": "20m", "B12": "20m"
        }
        
        filtered_files = []
        for band_file in band_files:
            band_id = self._extract_band_id(band_file)
            if band_id in band_resolutions and band_resolutions[band_id] == resolution:
                filtered_files.append(band_file)
        
        return filtered_files
    
    def _extract_band_id(self, filename: str) -> str:
        """Extract band ID from Sentinel-2 filename."""
        basename = os.path.basename(filename)
        parts = basename.split("_")
        
        for part in parts:
            if part.startswith("B") and len(part) >= 2:
                # Remove .jp2 extension if present
                band_id = part.replace(".jp2", "")
                return band_id
        
        return "Unknown"
    
    def _get_raster_crs(self, archive_path: str, band_file: str) -> CRS:
        """Get CRS from a raster band in ZIP archive."""
        vsi_path = f"/vsizip/{archive_path}/{band_file}"
        src_ds = gdal.Open(vsi_path)
        raster_crs_wkt = src_ds.GetProjection()
        src_ds = None
        return CRS.from_wkt(raster_crs_wkt)

    def _parse_metadata(self, archive_path: str) -> Dict[str, Any]:
        """Parse XML metadata from Sentinel-2 ZIP archive."""
        import xml.etree.ElementTree as ET

        metadata = {}

        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            # Find the main metadata file
            mtd_files = [
                f for f in zip_ref.namelist() if "MTD_MSIL" in f and f.endswith(".xml")
            ]

            if not mtd_files:
                raise FileNotFoundError("MTD_MSIL*.xml not found in ZIP archive")

            # Parse XML metadata
            with zip_ref.open(mtd_files[0]) as xml_file:
                tree = ET.parse(xml_file)
                root = tree.getroot()

                # Extract comprehensive metadata
                metadata['MTD_FILE'] = mtd_files[0]
                
                # Get product information
                for elem in root.iter():
                    tag_name = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
                    
                    # Key metadata fields
                    if tag_name in ['PRODUCT_URI', 'PROCESSING_LEVEL', 'SPACECRAFT_NAME', 
                                   'DATATAKE_IDENTIFIER', 'SENSING_TIME', 'PRODUCT_TYPE',
                                   'PROCESSING_BASELINE', 'GENERATION_TIME', 'CLOUD_COVERAGE_ASSESSMENT']:
                        if elem.text:
                            metadata[tag_name] = elem.text
                    
                    # Geometric information
                    elif tag_name in ['HORIZONTAL_CS_NAME', 'HORIZONTAL_CS_CODE']:
                        if elem.text:
                            metadata[tag_name] = elem.text
                
                # Add band count information
                band_count = 0
                for elem in root.iter():
                    if 'BAND_NAME' in elem.tag and elem.text:
                        band_count += 1
                metadata['ORIGINAL_BAND_COUNT'] = str(band_count)

        return metadata

    def _add_metadata_to_vrt(
        self,
        vrt_path: str,
        metadata: Dict[str, Any],
        band_files: List[str],
        source_path: str,
    ):
        """Add metadata to VRT file."""
        vrt_ds = gdal.Open(vrt_path, gdal.GA_Update)

        # Add XML metadata from Sentinel-2
        vrt_ds.SetMetadata(metadata, "S2_METADATA")

        # Add custom metadata
        custom_metadata = {
            "project": self.project_name,
            "source_archive": source_path,
            "band_count": str(len(band_files)),
            "satellite_type": "sentinel2",
            "resampled_bands": "all_bands_resampled_to_target_resolution",
            "processing_note": "All bands resampled to highest resolution grid"
        }
        vrt_ds.SetMetadata(custom_metadata)
        
        # Set band descriptions using extracted band IDs
        for i, band_file in enumerate(band_files, 1):
            if i <= vrt_ds.RasterCount:
                band = vrt_ds.GetRasterBand(i)
                band_id = self._extract_band_id(band_file)
                band.SetDescription(band_id)
                band.SetMetadataItem("DESCRIPTION", band_id)
                band.SetMetadataItem("BAND_NAME", band_id)

        vrt_ds = None


def create_vrt_builder(satellite_type: str, output_dir: str, **kwargs) -> VRTBuilder:
    """
    Factory function to create appropriate VRT builder.

    Parameters:
    -----------
    satellite_type : str
        Type of satellite ("landsat" or "sentinel2")
    output_dir : str
        Output directory for VRT files
    **kwargs
        Additional arguments for VRT builder

    Returns:
    --------
    VRTBuilder
        Appropriate VRT builder instance
    """
    if satellite_type.lower() == "landsat":
        return LandsatVRTBuilder(output_dir, **kwargs)
    elif satellite_type.lower() in ["sentinel2", "sentinel-2", "s2"]:
        return Sentinel2VRTBuilder(output_dir, **kwargs)
    else:
        raise ValueError(f"Unknown satellite type: {satellite_type}")


def batch_process_archives(
    archive_list: List[str],
    output_dir: str,
    bounds: Optional[gpd.GeoDataFrame] = None,
    satellite_type: Optional[str] = None,
    **kwargs,
):
    """
    Batch process multiple satellite archives to VRTs.

    Parameters:
    -----------
    archive_list : List[str]
        List of archive file paths
    output_dir : str
        Output directory for VRT files
    bounds : gpd.GeoDataFrame, optional
        Geographic bounds for cropping
    satellite_type : str, optional
        Force specific satellite type
    **kwargs
        Additional arguments for VRT builder
    """
    if not archive_list:
        print("No archives provided")
        return

    # Auto-detect satellite type if not provided
    if satellite_type is None:
        first_archive = Path(archive_list[0]).name.upper()
        if any(
            sat in first_archive for sat in ["LC08", "LC09", "LE07", "LT05", "LT04"]
        ):
            satellite_type = "landsat"
        elif any(sat in first_archive for sat in ["S2A", "S2B"]):
            satellite_type = "sentinel2"
        else:
            # Try to detect from file extension
            if archive_list[0].lower().endswith(".tar"):
                satellite_type = "landsat"
            elif archive_list[0].lower().endswith(".zip"):
                satellite_type = "sentinel2"
            else:
                raise ValueError(
                    "Cannot auto-detect satellite type. Please specify satellite_type parameter."
                )

    # Create VRT builder
    vrt_builder = create_vrt_builder(satellite_type, output_dir, **kwargs)

    # Process each archive
    created_vrts = []
    for archive_path in archive_list:
        try:
            vrt_path = vrt_builder.build_vrt(archive_path, bounds, **kwargs)
            created_vrts.append(vrt_path)
        except Exception as e:
            print(f"Failed to process {archive_path}: {e}")

    print(f"Successfully created {len(created_vrts)} VRT files")
    return created_vrts
