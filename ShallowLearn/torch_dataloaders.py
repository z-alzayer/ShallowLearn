"""
PyTorch DataLoader for unified Landsat and Sentinel-2 satellite imagery.
Handles band nomenclature unification, consistent sizing, and common band alignment.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from PIL import Image
import warnings

try:
    from .io.satellite_data import create_satellite_image, LandsatImage, Sentinel2Image
except ImportError:
    from ShallowLearn.io.satellite_data import create_satellite_image, LandsatImage, Sentinel2Image


class SatelliteDataset(Dataset):
    """
    PyTorch Dataset for unified Landsat and Sentinel-2 imagery.
    
    Handles:
    - Band nomenclature unification (B01 -> B1)
    - Image resizing for consistent dimensions
    - Common band alignment between satellite types
    - Data filtering and validation
    """
    
    # Band mapping between Sentinel-2 and Landsat nomenclature
    BAND_MAPPING = {
        # Sentinel-2 -> Landsat equivalent
        'B01': 'B1',   # Coastal/Aerosol
        'B02': 'B2',   # Blue
        'B03': 'B3',   # Green  
        'B04': 'B4',   # Red
        'B05': 'B5',   # Red Edge 1 -> NIR (approximate)
        'B06': 'B5',   # Red Edge 2 -> NIR (approximate) 
        'B07': 'B5',   # Red Edge 3 -> NIR (approximate)
        'B08': 'B5',   # NIR -> NIR
        'B8A': 'B5',   # NIR narrow -> NIR
        'B09': 'B9',   # Water vapour -> Cirrus (approximate)
        'B10': 'B9',   # Cirrus -> Cirrus
        'B11': 'B6',   # SWIR 1 -> SWIR 1
        'B12': 'B7',   # SWIR 2 -> SWIR 2
    }
    
    # Common bands available in both satellites (using Landsat nomenclature)
    COMMON_BANDS = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']
    
    def __init__(self, 
                 sentinel_paths: List[str] = None,
                 landsat_paths: List[str] = None,
                 sentinel_dir: str = None,
                 landsat_dir: str = None,
                 target_size: Tuple[int, int] = (512, 512),
                 bands: List[str] = None,
                 transform: Optional[callable] = None,
                 filter_invalid: bool = True,
                 min_valid_bands: int = 4):
        """
        Initialize the satellite dataset.
        
        Parameters:
        -----------
        sentinel_paths : List[str], optional
            List of Sentinel-2 file paths
        landsat_paths : List[str], optional  
            List of Landsat file paths
        sentinel_dir : str, optional
            Directory containing Sentinel-2 files (alternative to paths)
        landsat_dir : str, optional
            Directory containing Landsat files (alternative to paths)
        target_size : Tuple[int, int], default=(512, 512)
            Target image dimensions (height, width)
        bands : List[str], optional
            Specific bands to use (Landsat nomenclature). If None, uses COMMON_BANDS
        transform : callable, optional
            Additional transforms to apply
        filter_invalid : bool, default=True
            Whether to filter out images with insufficient valid bands
        min_valid_bands : int, default=4
            Minimum number of valid bands required if filtering
        """
        
        self.target_size = target_size
        self.bands = bands if bands is not None else self.COMMON_BANDS.copy()
        self.transform = transform
        self.filter_invalid = filter_invalid
        self.min_valid_bands = min_valid_bands
        
        # Collect file paths
        self.satellite_files = []
        
        # Add Sentinel-2 files
        if sentinel_paths:
            for path in sentinel_paths:
                self.satellite_files.append(('sentinel2', path))
        elif sentinel_dir:
            sentinel_dir = Path(sentinel_dir)
            for vrt_file in sentinel_dir.glob("*.vrt"):
                if any(sat in vrt_file.name.upper() for sat in ["S2A", "S2B"]):
                    self.satellite_files.append(('sentinel2', str(vrt_file)))
        
        # Add Landsat files  
        if landsat_paths:
            for path in landsat_paths:
                self.satellite_files.append(('landsat', path))
        elif landsat_dir:
            landsat_dir = Path(landsat_dir)
            for vrt_file in landsat_dir.glob("*.vrt"):
                if any(sat in vrt_file.name.upper() for sat in ["LC08", "LC09", "LE07", "LT05", "LT04"]):
                    self.satellite_files.append(('landsat', str(vrt_file)))
        
        if not self.satellite_files:
            raise ValueError("No satellite files found. Provide either file paths or directories.")
        
        # Filter invalid files if requested
        if self.filter_invalid:
            self._filter_valid_files()
        
        print(f"Dataset initialized with {len(self.satellite_files)} files")
        print(f"Target bands: {self.bands}")
        print(f"Target size: {self.target_size}")
    
    def _filter_valid_files(self):
        """Filter out files that don't meet minimum band requirements."""
        valid_files = []
        
        for sat_type, file_path in self.satellite_files:
            try:
                # Quick validation - load metadata only
                img = create_satellite_image(file_path)
                
                # Check how many requested bands are available
                unified_bands = self._get_unified_bands(img, sat_type)
                available_bands = sum(1 for band in self.bands if band in unified_bands)
                
                if available_bands >= self.min_valid_bands:
                    valid_files.append((sat_type, file_path))
                else:
                    warnings.warn(f"Skipping {file_path}: only {available_bands}/{len(self.bands)} bands available")
                    
            except Exception as e:
                warnings.warn(f"Skipping {file_path}: {str(e)}")
        
        self.satellite_files = valid_files
    
    def _get_unified_bands(self, img, sat_type: str) -> Dict[str, int]:
        """Get unified band mapping for the image."""
        if sat_type == 'sentinel2':
            # Convert Sentinel-2 bands to Landsat nomenclature
            unified = {}
            for s2_band, landsat_band in self.BAND_MAPPING.items():
                if img.has_band(s2_band):
                    unified[landsat_band] = img.band_order[s2_band]
            return unified
        else:
            # Landsat already uses correct nomenclature
            return {band: idx for band, idx in img.band_order.items() if img.has_band(band)}
    
    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """Resize image to target dimensions."""
        if image.shape[:2] == self.target_size:
            return image
        
        # Handle multi-band images
        if len(image.shape) == 3:
            resized_bands = []
            for band_idx in range(image.shape[2]):
                band = image[:, :, band_idx]
                # Use PIL for better interpolation
                pil_img = Image.fromarray(band.astype(np.float32))
                resized = pil_img.resize((self.target_size[1], self.target_size[0]), Image.BILINEAR)
                resized_bands.append(np.array(resized))
            return np.stack(resized_bands, axis=2)
        else:
            # Single band
            pil_img = Image.fromarray(image.astype(np.float32))
            resized = pil_img.resize((self.target_size[1], self.target_size[0]), Image.BILINEAR)
            return np.array(resized)
    
    def _extract_bands(self, img, sat_type: str) -> np.ndarray:
        """Extract and align requested bands from satellite image."""
        unified_bands = self._get_unified_bands(img, sat_type)
        
        # Extract requested bands in order
        band_arrays = []
        for band_name in self.bands:
            if band_name in unified_bands:
                if sat_type == 'sentinel2':
                    # Find the original Sentinel-2 band name
                    s2_band = None
                    for s2_name, landsat_name in self.BAND_MAPPING.items():
                        if landsat_name == band_name and img.has_band(s2_name):
                            s2_band = s2_name
                            break
                    
                    if s2_band:
                        band_data = img.get_band_data(s2_band)
                    else:
                        band_data = np.full((img.image.shape[0], img.image.shape[1]), np.nan)
                else:
                    # Landsat - direct mapping
                    band_data = img.get_band_data(band_name)
                
                if band_data is not None:
                    band_arrays.append(band_data)
                else:
                    # Create NaN placeholder
                    band_arrays.append(np.full((img.image.shape[0], img.image.shape[1]), np.nan))
            else:
                # Band not available - create NaN placeholder
                band_arrays.append(np.full((img.image.shape[0], img.image.shape[1]), np.nan))
        
        return np.stack(band_arrays, axis=2)
    
    def __len__(self) -> int:
        return len(self.satellite_files)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        """Get a single item from the dataset."""
        sat_type, file_path = self.satellite_files[idx]
        
        try:
            # Load satellite image
            img = create_satellite_image(file_path)
            
            # Extract and align bands
            image_data = self._extract_bands(img, sat_type)
            
            # Resize to target dimensions
            image_data = self._resize_image(image_data)
            
            # Convert to torch tensor and rearrange dimensions (H, W, C) -> (C, H, W)
            image_tensor = torch.from_numpy(image_data.astype(np.float32))
            image_tensor = image_tensor.permute(2, 0, 1)  # (C, H, W)
            
            # Apply additional transforms if provided
            if self.transform:
                image_tensor = self.transform(image_tensor)
            
            return {
                'image': image_tensor,
                'satellite_type': sat_type,
                'file_path': file_path,
                'bands': self.bands
            }
            
        except Exception as e:
            # Return a tensor of NaNs if loading fails
            warnings.warn(f"Failed to load {file_path}: {str(e)}")
            nan_tensor = torch.full((len(self.bands), *self.target_size), np.nan)
            
            return {
                'image': nan_tensor,
                'satellite_type': sat_type,
                'file_path': file_path,
                'bands': self.bands
            }
    
    def get_band_statistics(self) -> Dict[str, Dict[str, float]]:
        """Calculate statistics for each band across the dataset."""
        stats = {band: {'mean': [], 'std': [], 'min': [], 'max': []} for band in self.bands}
        
        for idx in range(len(self)):
            try:
                item = self[idx]
                image = item['image']  # Shape: (C, H, W)
                
                for band_idx, band_name in enumerate(self.bands):
                    band_data = image[band_idx].numpy()
                    
                    # Skip NaN values
                    valid_data = band_data[~np.isnan(band_data)]
                    
                    if len(valid_data) > 0:
                        stats[band_name]['mean'].append(np.mean(valid_data))
                        stats[band_name]['std'].append(np.std(valid_data))
                        stats[band_name]['min'].append(np.min(valid_data))
                        stats[band_name]['max'].append(np.max(valid_data))
                        
            except Exception as e:
                warnings.warn(f"Error calculating stats for index {idx}: {str(e)}")
        
        # Aggregate statistics
        aggregated_stats = {}
        for band_name in self.bands:
            if stats[band_name]['mean']:
                aggregated_stats[band_name] = {
                    'mean': np.mean(stats[band_name]['mean']),
                    'std': np.mean(stats[band_name]['std']),
                    'min': np.min(stats[band_name]['min']),
                    'max': np.max(stats[band_name]['max'])
                }
            else:
                aggregated_stats[band_name] = {
                    'mean': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan
                }
        
        return aggregated_stats


def create_satellite_dataloader(
    sentinel_dir: str = None,
    landsat_dir: str = None,
    sentinel_paths: List[str] = None,
    landsat_paths: List[str] = None,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    target_size: Tuple[int, int] = (512, 512),
    bands: List[str] = None,
    **dataset_kwargs
) -> DataLoader:
    """
    Create a PyTorch DataLoader for satellite imagery.
    
    Parameters:
    -----------
    sentinel_dir : str, optional
        Directory containing Sentinel-2 VRT files
    landsat_dir : str, optional
        Directory containing Landsat VRT files
    sentinel_paths : List[str], optional
        List of Sentinel-2 file paths
    landsat_paths : List[str], optional
        List of Landsat file paths
    batch_size : int, default=8
        Batch size for the DataLoader
    shuffle : bool, default=True
        Whether to shuffle the data
    num_workers : int, default=4
        Number of worker processes
    target_size : Tuple[int, int], default=(512, 512)
        Target image dimensions
    bands : List[str], optional
        Specific bands to use
    **dataset_kwargs
        Additional arguments for SatelliteDataset
    
    Returns:
    --------
    DataLoader
        PyTorch DataLoader for satellite imagery
    """
    
    dataset = SatelliteDataset(
        sentinel_paths=sentinel_paths,
        landsat_paths=landsat_paths,
        sentinel_dir=sentinel_dir,
        landsat_dir=landsat_dir,
        target_size=target_size,
        bands=bands,
        **dataset_kwargs
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_satellite_batch
    )


def collate_satellite_batch(batch: List[Dict]) -> Dict[str, Union[torch.Tensor, List]]:
    """
    Custom collate function for satellite data batches.
    
    Parameters:
    -----------
    batch : List[Dict]
        List of dataset items
        
    Returns:
    --------
    Dict
        Batched data with proper tensor stacking
    """
    
    # Stack images
    images = torch.stack([item['image'] for item in batch])
    
    # Collect metadata
    satellite_types = [item['satellite_type'] for item in batch]
    file_paths = [item['file_path'] for item in batch]
    bands = batch[0]['bands']  # Should be the same for all items
    
    return {
        'images': images,
        'satellite_types': satellite_types,
        'file_paths': file_paths,
        'bands': bands
    }