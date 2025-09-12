"""
File discovery utilities for satellite data processing.
"""

import re
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional


def find_files_in_directory(directory: str, max_files: int = 5) -> List[str]:
    """
    Find Sentinel-2 files in directory.
    
    Args:
        directory: Directory path to search
        max_files: Maximum number of files to return
        
    Returns:
        List of file paths as strings
    """
    directory = Path(directory)
    if not directory.exists():
        print(f"❌ Directory not found: {directory}")
        return []
    
    # Look for .SAFE directories and .zip files
    safe_files = list(directory.glob("*.SAFE"))[:max_files]
    zip_files = list(directory.glob("*.zip"))[:max_files]
    all_files = safe_files + zip_files
    
    print(f"Found {len(safe_files)} .SAFE and {len(zip_files)} .zip files (using {len(all_files)})")
    return [str(f) for f in all_files]


def find_matching_files_by_date(list1: List[str], list2: List[str]) -> List[Tuple[datetime, str, str]]:
    """
    Match files from two lists based on identical acquisition dates.
    Typically used for matching L1C and L2A Sentinel-2 files.
    
    Args:
        list1: First list of file paths
        list2: Second list of file paths
        
    Returns:
        List of tuples containing (date, file1, file2) for matching dates
    """
    def extract_date(file: str, prefix: str) -> Optional[datetime]:
        pattern = re.compile(rf"{prefix}(\d{{8}}T\d{{6}})")
        if match := pattern.search(file):
            return datetime.strptime(match.group(1), "%Y%m%dT%H%M%S")
        return None

    # Extract dates with validation
    l1_dates = [(extract_date(f, "MSIL1C_"), f) for f in list1]
    l2_dates = [(extract_date(f, "MSIL2A_"), f) for f in list2]

    # Filter invalid filenames and create date-keyed dictionaries
    valid_l1 = {d.date(): (d, f) for d, f in l1_dates if d}
    valid_l2 = {d.date(): (d, f) for d, f in l2_dates if d}

    # Find common dates
    common_dates = valid_l1.keys() & valid_l2.keys()

    return [
        (
            valid_l1[date][0],  # datetime object
            valid_l1[date][1],  # L1C path
            valid_l2[date][1],  # L2A path
        )
        for date in sorted(common_dates)
    ]


def find_landsat_files(directory: str, max_files: Optional[int] = None) -> List[str]:
    """
    Find Landsat files in directory.
    
    Args:
        directory: Directory path to search
        max_files: Maximum number of files to return (None for all)
        
    Returns:
        List of Landsat file paths as strings
    """
    directory = Path(directory)
    if not directory.exists():
        print(f"❌ Directory not found: {directory}")
        return []
    
    # Look for Landsat VRT files (common format)
    vrt_files = list(directory.glob("*cropped.vrt"))
    # Filter out problematic Landsat 7 files if needed
    vrt_files = [f for f in vrt_files if "LE07" not in f.name]
    
    if max_files is not None:
        vrt_files = vrt_files[:max_files]
    
    print(f"Found {len(vrt_files)} Landsat files")
    return [str(f) for f in vrt_files]


def detect_satellite_type(file_path: str) -> str:
    """
    Detect satellite type from file path.
    
    Args:
        file_path: Path to satellite file
        
    Returns:
        Satellite type string ('sentinel-2', 'landsat', 'unknown')
    """
    file_str = str(file_path).lower()
    
    if any(pattern in file_str for pattern in ['s2a_', 's2b_', 'msil1c', 'msil2a', '.safe']):
        return 'sentinel-2'
    elif any(pattern in file_str for pattern in ['lc08', 'lc09', 'le07', 'lt05', 'landsat']):
        return 'landsat'
    else:
        return 'unknown'


def process_reef_data(files, reef_gdf, reef_indices, data_type='L1C', buffer_meters=100):
    """
    Process satellite files for multiple reefs separately.
    
    Args:
        files: List of satellite file paths
        reef_gdf: GeoDataFrame containing reef polygons
        reef_indices: List of reef indices to process
        data_type: 'L1C' or 'L2A'
        buffer_meters: Buffer size for clipping
        
    Returns:
        Dict mapping reef names to lists of processed images
    """
    from ShallowLearn.io.satellite_data import Sentinel2Image
    from pathlib import Path
    
    reef_data = {}
    reef_names = reef_gdf.ORIG_NAME.to_list()
    
    for reef_idx in reef_indices:
        single_reef = reef_gdf.iloc[reef_idx:reef_idx+1].copy()
        reef_name = reef_names[reef_idx]
        reef_area = single_reef.iloc[0].Area
        
        print(f"\n📊 Processing {data_type} Reef: {reef_name} (Area: {reef_area:,.0f} m²)")
        
        reef_images = []
        for file_path in files:
            try:
                print(f"   Loading {data_type}: {Path(file_path).name}")
                s2_image = Sentinel2Image(
                    file_path,
                    load_all_bands=False,
                    clip_geometry=single_reef,
                    buffer_meters=buffer_meters
                )
                reef_images.append(s2_image)
                print(f"   ✅ Clipped to {s2_image.image.shape}")
                
            except Exception as e:
                print(f"   ❌ Failed: {e}")
        
        if reef_images:
            reef_data[reef_name] = reef_images
            print(f"   📋 {len(reef_images)} images processed for {reef_name}")
    
    return reef_data


def safe_filename(name: str) -> str:
    """Convert reef name to safe filename."""
    import re
    return re.sub(r'[^\w\-_.]', '_', str(name))