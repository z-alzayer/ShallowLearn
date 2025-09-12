"""
Input/Output module for ShallowLearn.
Handles loading and writing satellite data with VRT generation and metadata preservation.
"""

from .satellite_data import (
    SatelliteImage,
    LandsatImage, 
    Sentinel2Image,
    SatelliteImageCollection,
    LandsatImageCollection,
    Sentinel2ImageCollection,
    GeoTIFFImage,
    GeoTIFFCollection,
    LoadGeoTIFF,  # Backwards compatibility alias
    create_satellite_image,
    create_satellite_collection,
)

from .vrt_builder import (
    create_vrt_builder,
    batch_process_archives,
)

from .image_loader import (
    load_image,
    load_image_collection,
    load_img,  # Backwards compatibility alias
)

from .geotiff_compiler import (
    create_geotiff_compiler,
    batch_compile_geotiffs,
)

__all__ = [
    # Satellite data classes
    'SatelliteImage',
    'LandsatImage', 
    'Sentinel2Image',
    'SatelliteImageCollection',
    'LandsatImageCollection',
    'Sentinel2ImageCollection',
    
    # GeoTIFF classes
    'GeoTIFFImage',
    'GeoTIFFCollection',
    'LoadGeoTIFF',  # Backwards compatibility
    
    # Factory functions
    'create_satellite_image',
    'create_satellite_collection',
    
    # VRT builders
    'create_vrt_builder',
    'batch_process_archives',
    
    # High-level image loading
    'load_image',
    'load_image_collection', 
    'load_img',  # Backwards compatibility
    
    # GeoTIFF compilers
    'create_geotiff_compiler',
    'batch_compile_geotiffs',
]