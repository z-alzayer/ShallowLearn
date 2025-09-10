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
            
            # Check if bounds CRS matches raster CRS
            bounds_crs = str(bounds.crs) if bounds.crs else "EPSG:4326"
            if bounds_crs == raster_crs.to_string():
                # No transformation needed
                ulx, uly, lrx, lry = expanded_bounds[0], expanded_bounds[3], expanded_bounds[2], expanded_bounds[1]
            else:
                # Transform from bounds CRS to raster CRS
                ulx, uly, lrx, lry = self.transform_bounds(
                    expanded_bounds, bounds_crs, raster_crs.to_string()
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
        """Parse comprehensive MTL metadata from Landsat tar archive."""
        mtl_dict = {}
        stack = []

        with tarfile.open(archive_path, "r") as tar:
            mtl_member = next(
                (m for m in tar.getmembers() if m.name.endswith("MTL.txt")), None
            )
            if mtl_member is None:
                raise FileNotFoundError("MTL.txt not found in tar archive")

            # Enhanced MTL parsing - capture all fields with proper hierarchy
            mtl_file = tar.extractfile(mtl_member)
            current_group = None
            
            for line_bytes in mtl_file:
                line = line_bytes.decode("utf-8").strip()
                if not line or line == "END":
                    continue

                if line.startswith("GROUP = "):
                    group = line.split("=", 1)[1].strip()
                    stack.append(group)
                    current_group = group
                elif line.startswith("END_GROUP"):
                    if stack:
                        stack.pop()
                    current_group = stack[-1] if stack else None
                elif "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip('"')
                    
                    # Create hierarchical key with full path
                    full_key = ".".join(stack + [key])
                    mtl_dict[full_key] = value
                    
                    # Also create simplified keys for commonly used metadata
                    if stack:
                        simplified_key = f"{stack[-1]}.{key}"
                        mtl_dict[simplified_key] = value
                    
                    # Create top-level keys for critical metadata fields
                    critical_fields = [
                        'LANDSAT_SCENE_ID', 'LANDSAT_PRODUCT_ID', 'SPACECRAFT_ID', 'SENSOR_ID',
                        'DATE_ACQUIRED', 'SCENE_CENTER_TIME', 'WRS_PATH', 'WRS_ROW',
                        'CLOUD_COVER', 'SUN_AZIMUTH', 'SUN_ELEVATION', 'EARTH_SUN_DISTANCE',
                        'PROCESSING_SOFTWARE_VERSION', 'COLLECTION_NUMBER', 'COLLECTION_CATEGORY',
                        'TARGET_WRS_PATH', 'TARGET_WRS_ROW', 'NADIR_OFFNADIR', 'ROLL_ANGLE',
                        'THERMAL_LINES', 'REFLECTIVE_LINES', 'THERMAL_SAMPLES', 'REFLECTIVE_SAMPLES',
                        'GRID_CELL_SIZE_THERMAL', 'GRID_CELL_SIZE_REFLECTIVE', 'GRID_CELL_SIZE_PANCHROMATIC',
                        'ORIGIN_LAT', 'ORIGIN_LON', 'CORNER_UL_LAT_PRODUCT', 'CORNER_UL_LON_PRODUCT',
                        'CORNER_UR_LAT_PRODUCT', 'CORNER_UR_LON_PRODUCT', 'CORNER_LL_LAT_PRODUCT',
                        'CORNER_LL_LON_PRODUCT', 'CORNER_LR_LAT_PRODUCT', 'CORNER_LR_LON_PRODUCT',
                        'CORNER_UL_PROJECTION_X_PRODUCT', 'CORNER_UL_PROJECTION_Y_PRODUCT',
                        'CORNER_UR_PROJECTION_X_PRODUCT', 'CORNER_UR_PROJECTION_Y_PRODUCT',
                        'CORNER_LL_PROJECTION_X_PRODUCT', 'CORNER_LL_PROJECTION_Y_PRODUCT',
                        'CORNER_LR_PROJECTION_X_PRODUCT', 'CORNER_LR_PROJECTION_Y_PRODUCT',
                        'PANCHROMATIC_LINES', 'PANCHROMATIC_SAMPLES', 'FILE_DATE', 'STATION_ID',
                        'GROUND_CONTROL_POINTS_VERSION', 'GROUND_CONTROL_POINTS_MODEL',
                        'GEOMETRIC_RMSE_MODEL', 'GEOMETRIC_RMSE_MODEL_Y', 'GEOMETRIC_RMSE_MODEL_X'
                    ]
                    
                    if key in critical_fields:
                        mtl_dict[key] = value
                    
                    # Add band-specific metadata with enhanced structure
                    if key.startswith('FILE_NAME_BAND_'):
                        band_num = key.split('_')[-1]
                        mtl_dict[f'BAND_{band_num}_FILENAME'] = value
                    elif key.startswith('RADIANCE_MAXIMUM_BAND_'):
                        band_num = key.split('_')[-1]
                        mtl_dict[f'BAND_{band_num}_RADIANCE_MAX'] = value
                    elif key.startswith('RADIANCE_MINIMUM_BAND_'):
                        band_num = key.split('_')[-1]
                        mtl_dict[f'BAND_{band_num}_RADIANCE_MIN'] = value
                    elif key.startswith('REFLECTANCE_MAXIMUM_BAND_'):
                        band_num = key.split('_')[-1]
                        mtl_dict[f'BAND_{band_num}_REFLECTANCE_MAX'] = value
                    elif key.startswith('REFLECTANCE_MINIMUM_BAND_'):
                        band_num = key.split('_')[-1]
                        mtl_dict[f'BAND_{band_num}_REFLECTANCE_MIN'] = value

            # Add archive-level metadata
            mtl_dict['MTL_FILE_PATH'] = mtl_member.name
            mtl_dict['ARCHIVE_PATH'] = archive_path
            mtl_dict['TOTAL_MTL_FIELDS'] = str(len(mtl_dict))

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
        Optimized to do everything in a single pass through the ZIP file.

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
        vrt_base = archive_path.stem
        
        # Determine output path based on whether we're cropping
        if bounds is not None:
            final_vrt_path = self.output_dir / f"{vrt_base}_cropped_{target_resolution}.vrt"
        else:
            final_vrt_path = self.output_dir / f"{vrt_base}_{target_resolution}.vrt"

        # Single pass through ZIP file to get everything we need
        band_files = []
        metadata = {}
        
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            all_files = zip_ref.namelist()
            
            # Get band files
            band_files = [f for f in all_files 
                         if "IMG_DATA" in f and f.endswith(".jp2") and "TCI" not in f]
            band_files.sort(key=self._extract_band_number)
            
            if not band_files:
                raise RuntimeError(f"No JP2 files found in {archive_path}")
            
            # Quick metadata extraction from XML (only essential fields)
            mtd_files = [f for f in all_files if "MTD_MSIL" in f and f.endswith(".xml")]
            if mtd_files:
                import xml.etree.ElementTree as ET
                with zip_ref.open(mtd_files[0]) as xml_file:
                    tree = ET.parse(xml_file)
                    root = tree.getroot()
                    
                    # Only extract critical fields needed for analysis
                    for elem in root.iter():
                        tag_name = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
                        if tag_name in ['PRODUCT_URI', 'PROCESSING_LEVEL', 'SPACECRAFT_NAME', 
                                       'CLOUD_COVERAGE_ASSESSMENT', 'GENERATION_TIME',
                                       'DATATAKE_IDENTIFIER', 'PRODUCT_TYPE', 'AOT_RETRIEVAL_METHOD']:
                            if elem.text:
                                metadata[tag_name] = elem.text
                    
                    # For L2A, also check for AOT_RETRIEVAL_METHOD in Quality_Indicators_Info
                    if 'L2A' in str(archive_path):
                        for qi in root.findall('.//{*}Quality_Indicators_Info'):
                            aot_elem = qi.find('.//{*}AOT_RETRIEVAL_METHOD')
                            if aot_elem is not None and aot_elem.text:
                                metadata['AOT_RETRIEVAL_METHOD'] = aot_elem.text

        # Get the highest resolution bands to use as reference grid
        reference_bands = self._filter_by_resolution(band_files, target_resolution)
        if not reference_bands:
            raise RuntimeError(f"No reference bands found for resolution {target_resolution}")

        # Build /vsizip/ paths for ALL bands
        vsi_paths = [f"/vsizip/{archive_path}/{band_file}" for band_file in band_files]
        
        # Get reference band info - just open once quickly
        ref_vsi_path = f"/vsizip/{archive_path}/{reference_bands[0]}"
        ref_ds = gdal.Open(ref_vsi_path)
        if not ref_ds:
            raise RuntimeError(f"Could not open reference band: {ref_vsi_path}")
            
        target_transform = ref_ds.GetGeoTransform()
        target_width = ref_ds.RasterXSize
        target_height = ref_ds.RasterYSize
        raster_crs_wkt = ref_ds.GetProjection()
        ref_ds = None
        
        from pyproj import CRS
        raster_crs = CRS.from_wkt(raster_crs_wkt)
        
        # If we have bounds, create cropped VRT directly
        if bounds is not None:
            # Expand and transform bounds
            expanded_bounds = self.expand_bounds(
                bounds.total_bounds, n_pixels, pixel_size
            )
            
            # Transform bounds if needed
            bounds_crs = str(bounds.crs) if bounds.crs else "EPSG:4326"
            if bounds_crs == str(raster_crs):
                ulx, uly, lrx, lry = expanded_bounds[0], expanded_bounds[3], expanded_bounds[2], expanded_bounds[1]
            else:
                ulx, uly, lrx, lry = self.transform_bounds(
                    expanded_bounds, bounds_crs, str(raster_crs)
                )
            
            # Create VRT with cropping and resampling in one step
            vrt_options = gdal.BuildVRTOptions(
                resolution='user',
                xRes=abs(target_transform[1]),  # pixel width
                yRes=abs(target_transform[5]),  # pixel height
                outputBounds=[ulx, lry, lrx, uly],  # cropped bounds
                targetAlignedPixels=True,
                separate=True,
                resampleAlg='bilinear'
            )
        else:
            # Create full VRT with resampling
            vrt_options = gdal.BuildVRTOptions(
                resolution='user',
                xRes=abs(target_transform[1]),
                yRes=abs(target_transform[5]),
                outputBounds=[target_transform[0], 
                             target_transform[3] + target_height * target_transform[5],
                             target_transform[0] + target_width * target_transform[1],
                             target_transform[3]],
                targetAlignedPixels=True,
                separate=True,
                resampleAlg='bilinear'
            )
        
        # Create the VRT directly with cropping if bounds provided
        vrt_ds = gdal.BuildVRT(str(final_vrt_path), vsi_paths, options=vrt_options)
        vrt_ds = None
        
        # Add metadata to VRT (lightweight - no subdataset opening)  
        self._add_metadata_to_vrt_fast(
            str(final_vrt_path), metadata, band_files, str(archive_path)
        )
        
        print(f"Created: {final_vrt_path}")
        return str(final_vrt_path)
    
    def _add_metadata_to_vrt_fast(
        self,
        vrt_path: str,
        metadata: Dict[str, Any],
        band_files: List[str],
        source_path: str,
    ):
        """Add metadata to VRT file - lightweight version without subdataset opening."""
        vrt_ds = gdal.Open(vrt_path, gdal.GA_Update)

        # Add essential metadata only (no RASTER_TAGS domain)
        if metadata:
            vrt_ds.SetMetadata(metadata, "S2_METADATA")

        # Add custom metadata
        custom_metadata = {
            "project": self.project_name,
            "source_archive": source_path,
            "band_count": str(len(band_files)),
            "satellite_type": "sentinel2",
            "processing_note": "Optimized single-pass VRT creation"
        }
        vrt_ds.SetMetadata(custom_metadata)
        
        # Set band descriptions
        for i, band_file in enumerate(band_files, 1):
            if i <= vrt_ds.RasterCount:
                band = vrt_ds.GetRasterBand(i)
                band_id = self._extract_band_id(band_file)
                band.SetDescription(band_id)
                band.SetMetadataItem("BAND_NAME", band_id)

        vrt_ds = None


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
    
    def _parse_metadata_lazy(self, archive_path: str, required_fields: List[str] = None) -> Dict[str, Any]:
        """
        Parse only required metadata fields from Sentinel-2 ZIP archive.
        This is a performance-optimized version that only extracts what's needed.
        
        Parameters:
        -----------
        archive_path : str
            Path to ZIP archive
        required_fields : List[str], optional
            List of specific fields to extract. If None, extracts minimal set.
            
        Returns:
        --------
        Dict[str, Any]
            Dictionary containing only requested metadata
        """
        import xml.etree.ElementTree as ET
        
        # Default minimal fields if none specified
        if required_fields is None:
            required_fields = ['PRODUCT_URI', 'PROCESSING_LEVEL', 'SPACECRAFT_NAME', 
                             'CLOUD_COVERAGE_ASSESSMENT', 'AOT_RETRIEVAL_METHOD']
        
        metadata = {}
        
        # Quick XML extraction for essential fields only
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            mtd_files = [f for f in zip_ref.namelist() if "MTD_MSIL" in f and f.endswith(".xml")]
            
            if mtd_files:
                with zip_ref.open(mtd_files[0]) as xml_file:
                    tree = ET.parse(xml_file)
                    root = tree.getroot()
                    metadata['MTD_FILE'] = mtd_files[0]
                    
                    # Only search for required fields
                    for elem in root.iter():
                        tag_name = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
                        if tag_name in required_fields and elem.text:
                            metadata[tag_name] = elem.text
                            
                    # For L2A, check for AOT_RETRIEVAL_METHOD in Quality_Indicators
                    if 'AOT_RETRIEVAL_METHOD' in required_fields:
                        # Try to find it in the XML structure
                        for qi in root.findall('.//{*}Quality_Indicators_Info'):
                            for child in qi:
                                if 'AOT_RETRIEVAL_METHOD' in child.tag:
                                    metadata['AOT_RETRIEVAL_METHOD'] = child.text
        
        return metadata
    
    def extract_metadata_from_vrt(self, vrt_path: str) -> Dict[str, Any]:
        """
        Extract metadata stored in VRT file.
        Much faster than parsing original archives.
        
        Parameters:
        -----------
        vrt_path : str
            Path to VRT file
            
        Returns:
        --------
        Dict[str, Any]
            Metadata dictionary
        """
        metadata = {}
        
        try:
            vrt_ds = gdal.Open(vrt_path, gdal.GA_ReadOnly)
            if vrt_ds:
                # Get S2_METADATA domain
                s2_metadata = vrt_ds.GetMetadata("S2_METADATA")
                if s2_metadata:
                    metadata.update(s2_metadata)
                
                # Get default domain metadata
                default_metadata = vrt_ds.GetMetadata()
                if default_metadata:
                    metadata.update(default_metadata)
                
                # Get RASTER_TAGS if stored separately
                raster_tags = vrt_ds.GetMetadata("RASTER_TAGS")
                if raster_tags:
                    metadata['RASTER_TAGS'] = raster_tags
                
                vrt_ds = None
        except Exception as e:
            print(f"Warning: Could not extract metadata from VRT: {e}")
        
        return metadata

    def _parse_metadata(self, archive_path: str) -> Dict[str, Any]:
        """Parse comprehensive metadata from Sentinel-2 ZIP archive using subdatasets approach."""
        import xml.etree.ElementTree as ET
        import rasterio

        metadata = {}
        
        # First, get metadata from subdatasets (like original LoadSentinel2L1C)
        try:
            # Find the main metadata file for subdatasets
            with zipfile.ZipFile(archive_path, "r") as zip_ref:
                mtd_files = [f for f in zip_ref.namelist() if "MTD_MSIL" in f and f.endswith(".xml")]
                if not mtd_files:
                    raise FileNotFoundError("MTD_MSIL*.xml not found in ZIP archive")
                
                mtd_file_path = mtd_files[0]
                zip_path = f"/vsizip/{archive_path}"
                full_mtd_path = os.path.join(zip_path, mtd_file_path)
                
                # Open subdatasets using the MTD file (original approach)
                with rasterio.open(full_mtd_path) as dataset:
                    subdatasets = dataset.subdatasets
                    
                    if subdatasets:
                        # Extract metadata from first subdataset
                        with rasterio.open(subdatasets[0]) as first_array:
                            # Get comprehensive rasterio metadata
                            raster_tags = first_array.tags()
                            raster_profile = first_array.profile
                            raster_metadata = first_array.meta
                            raster_offsets = first_array.offsets
                            raster_bounds = first_array.bounds
                            
                            # Add rasterio metadata to our metadata dict
                            metadata.update({
                                'RASTER_TAGS': raster_tags,
                                'RASTER_PROFILE_WIDTH': str(raster_profile.get('width', '')),
                                'RASTER_PROFILE_HEIGHT': str(raster_profile.get('height', '')),
                                'RASTER_PROFILE_COUNT': str(raster_profile.get('count', '')),
                                'RASTER_PROFILE_DTYPE': str(raster_profile.get('dtype', '')),
                                'RASTER_PROFILE_CRS': str(raster_profile.get('crs', '')),
                                'RASTER_BOUNDS_LEFT': str(raster_bounds.left),
                                'RASTER_BOUNDS_BOTTOM': str(raster_bounds.bottom),
                                'RASTER_BOUNDS_RIGHT': str(raster_bounds.right),
                                'RASTER_BOUNDS_TOP': str(raster_bounds.top),
                                'RASTER_TRANSFORM': str(raster_profile.get('transform', '')),
                                'SUBDATASETS_COUNT': str(len(subdatasets))
                            })
                            
                            # Add band offsets if available
                            if raster_offsets:
                                for i, offset in enumerate(raster_offsets):
                                    if offset is not None:
                                        metadata[f'BAND_{i+1}_OFFSET'] = str(offset)
                            
                            # Add additional raster tags as individual metadata items
                            for key, value in raster_tags.items():
                                metadata[f'TAG_{key}'] = str(value)

        except Exception as e:
            print(f"Warning: Could not extract subdataset metadata: {e}")

        # Now extract XML metadata (enhanced approach)
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            mtd_files = [f for f in zip_ref.namelist() if "MTD_MSIL" in f and f.endswith(".xml")]
            
            if mtd_files:
                with zip_ref.open(mtd_files[0]) as xml_file:
                    tree = ET.parse(xml_file)
                    root = tree.getroot()

                    # Enhanced XML metadata extraction - capture more fields
                    metadata['MTD_FILE'] = mtd_files[0]
                    
                    # Extract all text elements with meaningful content
                    for elem in root.iter():
                        tag_name = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
                        
                        # Comprehensive list of important metadata fields
                        important_fields = [
                            'PRODUCT_URI', 'PROCESSING_LEVEL', 'SPACECRAFT_NAME', 
                            'DATATAKE_IDENTIFIER', 'SENSING_TIME', 'PRODUCT_TYPE',
                            'PROCESSING_BASELINE', 'GENERATION_TIME', 'CLOUD_COVERAGE_ASSESSMENT',
                            'HORIZONTAL_CS_NAME', 'HORIZONTAL_CS_CODE', 'GEOMETRIC_QUALITY_FLAG',
                            'GENERAL_QUALITY_FLAG', 'RADIOMETRIC_QUALITY_FLAG', 'SENSOR_QUALITY_FLAG',
                            'MEAN_SUN_ANGLE', 'MEAN_VIEWING_INCIDENCE_ANGLE', 'MEAN_VIEWING_AZIMUTH_ANGLE',
                            'QUANTIFICATION_VALUE', 'REFLECTANCE_CONVERSION', 'U', 'SPECIAL_VALUE_NODATA',
                            'SPECIAL_VALUE_SATURATED', 'TILE_ID', 'DATASTRIP_ID', 'ARCHIVING_CENTRE',
                            'ARCHIVING_TIME', 'DEGRADED_ANC_DATA_PERCENTAGE', 'DEGRADED_MSI_DATA_PERCENTAGE'
                        ]
                        
                        if tag_name in important_fields and elem.text:
                            metadata[tag_name] = elem.text
                        
                        # Also capture band-specific information
                        elif 'BAND_NAME' in tag_name and elem.text:
                            parent = elem.getparent()
                            if parent is not None:
                                band_info_key = f"BAND_INFO_{elem.text}"
                                for child in parent:
                                    child_tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
                                    if child.text and child_tag != 'BAND_NAME':
                                        metadata[f"{band_info_key}_{child_tag}"] = child.text

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

        # Extract RASTER_TAGS if present and store separately
        raster_tags = metadata.get('RASTER_TAGS', {})
        if raster_tags:
            # Store RASTER_TAGS in a dedicated domain for easy retrieval
            if isinstance(raster_tags, dict):
                vrt_ds.SetMetadata(raster_tags, "RASTER_TAGS")
        
        # Create a clean metadata dict without RASTER_TAGS for S2_METADATA domain
        clean_metadata = {k: v for k, v in metadata.items() if k != 'RASTER_TAGS'}
        
        # Add XML metadata from Sentinel-2
        vrt_ds.SetMetadata(clean_metadata, "S2_METADATA")

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
