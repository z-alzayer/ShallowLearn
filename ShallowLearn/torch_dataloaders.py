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
                 labels_dir: str = None,
                 target_size: Tuple[int, int] = (512, 512),
                 bands: List[str] = None,
                 transform: Optional[callable] = None,
                 auto_find_common_bands: bool = True,
                 apply_scaling: bool = True):
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
        labels_dir : str, optional
            Directory containing pre-computed labels (.npy files) matching image filenames
        target_size : Tuple[int, int], default=(512, 512)
            Target image dimensions (height, width)
        bands : List[str], optional
            Specific bands to use (Landsat nomenclature). If None, auto-discovers common bands
        transform : callable, optional
            Additional transforms to apply
        auto_find_common_bands : bool, default=True
            Whether to automatically find common bands across all files
        apply_scaling : bool, default=True
            Whether to apply satellite-specific scaling (Sentinel-2: /10000, Landsat: metadata-based)
        """
        
        self.target_size = target_size
        self.transform = transform
        self.auto_find_common_bands = auto_find_common_bands
        self.apply_scaling = apply_scaling
        self.labels_dir = Path(labels_dir) if labels_dir else None
        
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
        
        # Auto-discover common bands if requested
        if self.auto_find_common_bands or bands is None:
            self.bands = self._find_common_bands()
            if not self.bands:
                raise ValueError("No common bands found across all files")
        else:
            self.bands = bands
        
        print(f"Dataset initialized with {len(self.satellite_files)} files")
        print(f"Common bands found: {self.bands}")
        print(f"Target size: {self.target_size}")
    
    def _find_common_bands(self) -> List[str]:
        """Find bands that are available in ALL files."""
        print("Discovering common bands across all files...")
        
        all_available_bands = []
        
        for sat_type, file_path in self.satellite_files:
            try:
                img = create_satellite_image(file_path)
                unified_bands = self._get_unified_bands(img, sat_type)
                # Only include bands that actually have data (not placeholders)
                available_bands = list(unified_bands.keys())
                all_available_bands.append(set(available_bands))
                print(f"  {Path(file_path).name}: {len(available_bands)} bands - {available_bands}")
                
            except Exception as e:
                warnings.warn(f"Error reading {file_path}: {str(e)}")
                continue
        
        if not all_available_bands:
            return []
        
        # Find intersection of all band sets
        common_bands = set.intersection(*all_available_bands)
        common_bands_list = sorted(list(common_bands), key=lambda x: int(x[1:]) if x[1:].isdigit() else 999)
        
        print(f"Common bands across all files: {common_bands_list}")
        return common_bands_list
    
    def _get_unified_bands(self, img, sat_type: str) -> Dict[str, int]:
        """Get unified band mapping for the image."""
        if sat_type == 'sentinel2':
            # Convert Sentinel-2 bands to Landsat nomenclature
            unified = {}
            for s2_band, landsat_band in self.BAND_MAPPING.items():
                if img.has_band(s2_band):
                    # If this Landsat band is already mapped, prioritize certain S2 bands
                    if landsat_band not in unified:
                        unified[landsat_band] = s2_band  # Store S2 band name, not index
                    else:
                        # Priority: B08 > B05 > B06 > B07 for NIR (B5)
                        if landsat_band == 'B5':
                            priority = {'B08': 1, 'B05': 2, 'B06': 3, 'B07': 4, 'B8A': 5}
                            current_s2 = unified[landsat_band]
                            if priority.get(s2_band, 6) < priority.get(current_s2, 6):
                                unified[landsat_band] = s2_band
            return unified
        else:
            # Landsat already uses correct nomenclature
            return {band: band for band in img.band_order.keys() if img.has_band(band)}
    
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
        """Extract and align requested bands from satellite image (only existing bands, no NaNs)."""
        band_arrays = []
        unified_bands = self._get_unified_bands(img, sat_type)
        
        for band_name in self.bands:
            if band_name in unified_bands:
                if sat_type == 'sentinel2':
                    # Get the S2 band name that maps to this Landsat band
                    s2_band = unified_bands[band_name]
                    band_data = img.get_band_data(s2_band)
                    if band_data is not None:
                        band_arrays.append(band_data)
                    else:
                        raise ValueError(f"Failed to extract Sentinel-2 band {s2_band} -> {band_name}")
                else:
                    # Landsat - direct mapping
                    band_data = img.get_band_data(band_name)
                    if band_data is not None:
                        band_arrays.append(band_data)
                    else:
                        raise ValueError(f"Failed to extract Landsat band {band_name}")
            else:
                raise ValueError(f"Band {band_name} not available in {sat_type} image")
        
        image_data = np.stack(band_arrays, axis=2)
        
        # Apply scaling if requested
        if self.apply_scaling:
            image_data = self._apply_scaling(image_data, img, sat_type)
        
        return image_data
    
    def _apply_scaling(self, image_data: np.ndarray, img, sat_type: str) -> np.ndarray:
        """Apply satellite-specific scaling to image data."""
        if sat_type == 'sentinel2':
            # Sentinel-2 scaling: divide by 10000 to get reflectance values
            return image_data / 10000.0
        else:
            # Landsat scaling: typically already in reflectance or can use metadata
            # For now, assume already scaled or apply simple normalization
            return image_data / 65535.0  # Assuming 16-bit data
    
    def _get_label_path(self, image_path: str) -> Optional[str]:
        """Find matching label file for the image."""
        if not self.labels_dir:
            return None
        
        # Extract filename without extension
        image_name = Path(image_path).stem
        
        # Look for matching .npy file
        label_path = self.labels_dir / f"{image_name}.npy"
        
        if label_path.exists():
            return str(label_path)
        
        return None
    
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
            
            # Load labels if available
            label_tensor = None
            label_path = self._get_label_path(file_path)
            if label_path:
                try:
                    labels = np.load(label_path)
                    label_tensor = torch.from_numpy(labels.astype(np.float32))
                except Exception as e:
                    warnings.warn(f"Failed to load labels from {label_path}: {str(e)}")
            
            result = {
                'image': image_tensor,
                'satellite_type': sat_type,
                'file_path': file_path,
                'bands': self.bands
            }
            
            if label_tensor is not None:
                result['labels'] = label_tensor
            
            return result
            
        except Exception as e:
            # Re-raise the exception since we want to catch data issues early
            raise RuntimeError(f"Failed to load {file_path}: {str(e)}") from e
    
    def get_band_statistics(self) -> Dict[str, Dict[str, float]]:
        """Calculate statistics for each band across the dataset."""
        stats = {band: {'mean': [], 'std': [], 'min': [], 'max': []} for band in self.bands}
        
        for idx in range(len(self)):
            try:
                item = self[idx]
                image = item['image']  # Shape: (C, H, W)
                
                for band_idx, band_name in enumerate(self.bands):
                    band_data = image[band_idx].numpy()
                    
                    # All data should be valid (no NaNs)
                    stats[band_name]['mean'].append(np.mean(band_data))
                    stats[band_name]['std'].append(np.std(band_data))
                    stats[band_name]['min'].append(np.min(band_data))
                    stats[band_name]['max'].append(np.max(band_data))
                        
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
                # Should not happen with common bands approach
                raise ValueError(f"No statistics available for band {band_name}")
        
        return aggregated_stats


def create_satellite_dataloader(
    sentinel_dir: str = None,
    landsat_dir: str = None,
    sentinel_paths: List[str] = None,
    landsat_paths: List[str] = None,
    labels_dir: str = None,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    target_size: Tuple[int, int] = (512, 512),
    bands: List[str] = None,
    auto_find_common_bands: bool = True,
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
    labels_dir : str, optional
        Directory containing pre-computed labels (.npy files)
    batch_size : int, default=8
        Batch size for the DataLoader
    shuffle : bool, default=True
        Whether to shuffle the data
    num_workers : int, default=4
        Number of worker processes
    target_size : Tuple[int, int], default=(512, 512)
        Target image dimensions
    bands : List[str], optional
        Specific bands to use (if None, auto-discovers common bands)
    auto_find_common_bands : bool, default=True
        Whether to automatically find common bands across all files
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
        labels_dir=labels_dir,
        target_size=target_size,
        bands=bands,
        auto_find_common_bands=auto_find_common_bands,
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
    
    result = {
        'images': images,
        'satellite_types': satellite_types,
        'file_paths': file_paths,
        'bands': bands
    }
    
    # Stack labels if available
    if 'labels' in batch[0]:
        labels = torch.stack([item['labels'] for item in batch])
        result['labels'] = labels
    
    return result