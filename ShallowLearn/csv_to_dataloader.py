"""
Utilities to extract file paths from CSV decisions and create dataloaders.
"""

import pandas as pd
from typing import List, Dict, Tuple, Optional
from pathlib import Path


def load_decisions_csv(csv_path: str) -> pd.DataFrame:
    """Load the decisions CSV file."""
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ Loaded {len(df)} decisions from {csv_path}")
        return df
    except Exception as e:
        raise FileNotFoundError(f"Error loading CSV: {e}")


def extract_paths_by_status(df: pd.DataFrame, 
                          status: str = 'train',
                          satellite_type: Optional[str] = None) -> Dict[str, List[str]]:
    """
    Extract file paths grouped by satellite type for a given status.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame from decisions CSV
    status : str
        Status to filter by ('train', 'validation', 'skip', 'undecided')
    satellite_type : str, optional
        Filter by satellite type ('sentinel2', 'landsat', or None for both)
    
    Returns:
    --------
    Dict[str, List[str]]
        Dictionary with 'sentinel2' and 'landsat' keys containing file paths
    """
    
    # Filter by status
    filtered_df = df[df['status'] == status].copy()
    
    if len(filtered_df) == 0:
        print(f"⚠️  No files found with status '{status}'")
        return {'sentinel2': [], 'landsat': []}
    
    # Filter by satellite type if specified
    if satellite_type:
        filtered_df = filtered_df[filtered_df['satellite_type'] == satellite_type]
        if len(filtered_df) == 0:
            print(f"⚠️  No {satellite_type} files found with status '{status}'")
    
    # Group by satellite type
    sentinel_paths = filtered_df[filtered_df['satellite_type'] == 'sentinel2']['file_path'].tolist()
    landsat_paths = filtered_df[filtered_df['satellite_type'] == 'landsat']['file_path'].tolist()
    
    print(f"📊 Status '{status}' - Sentinel-2: {len(sentinel_paths)}, Landsat: {len(landsat_paths)}")
    
    return {
        'sentinel2': sentinel_paths,
        'landsat': landsat_paths
    }


def create_train_val_split(csv_path: str) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Create training and validation splits from CSV decisions.
    
    Parameters:
    -----------
    csv_path : str
        Path to decisions CSV file
    
    Returns:
    --------
    Tuple[Dict[str, List[str]], Dict[str, List[str]]]
        (train_paths, val_paths) where each dict has 'sentinel2' and 'landsat' keys
    """
    
    df = load_decisions_csv(csv_path)
    
    # Extract paths for training and validation
    train_paths = extract_paths_by_status(df, 'train')
    val_paths = extract_paths_by_status(df, 'validation')
    
    # Summary
    total_train = len(train_paths['sentinel2']) + len(train_paths['landsat'])
    total_val = len(val_paths['sentinel2']) + len(val_paths['landsat'])
    
    print(f"\n📋 SUMMARY:")
    print(f"   Training: {total_train} files ({len(train_paths['sentinel2'])} S2, {len(train_paths['landsat'])} Landsat)")
    print(f"   Validation: {total_val} files ({len(val_paths['sentinel2'])} S2, {len(val_paths['landsat'])} Landsat)")
    
    return train_paths, val_paths


def verify_file_paths(paths_dict: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    Verify that all file paths exist and return only valid ones.
    
    Parameters:
    -----------
    paths_dict : Dict[str, List[str]]
        Dictionary with satellite types as keys and file paths as values
    
    Returns:
    --------
    Dict[str, List[str]]
        Dictionary with only existing file paths
    """
    
    verified_paths = {'sentinel2': [], 'landsat': []}
    
    for sat_type, paths in paths_dict.items():
        valid_paths = []
        missing_paths = []
        
        for path in paths:
            if Path(path).exists():
                valid_paths.append(path)
            else:
                missing_paths.append(path)
        
        verified_paths[sat_type] = valid_paths
        
        if missing_paths:
            print(f"⚠️  {sat_type}: {len(missing_paths)} missing files")
            for missing in missing_paths[:3]:  # Show first 3
                print(f"     Missing: {Path(missing).name}")
            if len(missing_paths) > 3:
                print(f"     ... and {len(missing_paths) - 3} more")
        
        print(f"✅ {sat_type}: {len(valid_paths)}/{len(paths)} files exist")
    
    return verified_paths


def create_dataloader_from_csv(csv_path: str,
                              status: str = 'train',
                              labels_dir: str = None,
                              verify_files: bool = True,
                              **dataloader_kwargs) -> 'DataLoader':
    """
    Create a dataloader from CSV decisions.
    
    Parameters:
    -----------
    csv_path : str
        Path to decisions CSV file
    status : str
        Status to use ('train', 'validation')
    labels_dir : str, optional
        Directory containing labels
    verify_files : bool
        Whether to verify file existence before creating dataloader
    **dataloader_kwargs
        Additional arguments for create_satellite_dataloader
    
    Returns:
    --------
    DataLoader
        PyTorch DataLoader with selected files
    """
    
    from ShallowLearn.torch_dataloaders import create_satellite_dataloader
    
    # Load decisions and extract paths
    df = load_decisions_csv(csv_path)
    paths_dict = extract_paths_by_status(df, status)
    
    # Verify files exist if requested
    if verify_files:
        paths_dict = verify_file_paths(paths_dict)
    
    # Check if we have any files
    total_files = len(paths_dict['sentinel2']) + len(paths_dict['landsat'])
    if total_files == 0:
        raise ValueError(f"No valid files found for status '{status}'")
    
    # Create dataloader
    sentinel_paths = paths_dict['sentinel2'] if paths_dict['sentinel2'] else None
    landsat_paths = paths_dict['landsat'] if paths_dict['landsat'] else None
    
    print(f"\n🚀 Creating dataloader for '{status}' with {total_files} files...")
    
    dataloader = create_satellite_dataloader(
        sentinel_paths=sentinel_paths,
        landsat_paths=landsat_paths,
        labels_dir=labels_dir,
        **dataloader_kwargs
    )
    
    print(f"✅ Dataloader created successfully!")
    return dataloader


def show_csv_summary(csv_path: str):
    """Show a summary of the CSV decisions."""
    df = load_decisions_csv(csv_path)
    
    print(f"\n📊 CSV SUMMARY: {Path(csv_path).name}")
    print("="*50)
    
    # Status counts
    status_counts = df['status'].value_counts()
    print("📋 Status Distribution:")
    for status, count in status_counts.items():
        print(f"   {status}: {count}")
    
    # Satellite type counts
    print(f"\n🛰️  Satellite Distribution:")
    sat_counts = df['satellite_type'].value_counts()
    for sat_type, count in sat_counts.items():
        print(f"   {sat_type}: {count}")
    
    # Cross-tabulation
    print(f"\n📊 Status by Satellite Type:")
    crosstab = pd.crosstab(df['satellite_type'], df['status'])
    print(crosstab)
    
    # Files with notes
    notes_count = df[df['notes'] != '']['notes'].count()
    print(f"\n📝 Files with notes: {notes_count}")
    
    return df


# Convenience functions for common use cases
def create_train_dataloader(csv_path: str, labels_dir: str = None, **kwargs):
    """Create training dataloader from CSV."""
    return create_dataloader_from_csv(csv_path, 'train', labels_dir, **kwargs)


def create_val_dataloader(csv_path: str, labels_dir: str = None, **kwargs):
    """Create validation dataloader from CSV."""
    return create_dataloader_from_csv(csv_path, 'validation', labels_dir, **kwargs)


def get_file_lists_for_manual_dataloader(csv_path: str, status: str = 'train'):
    """
    Get file lists that can be passed manually to create_satellite_dataloader.
    
    Returns:
    --------
    Tuple[List[str], List[str]]
        (sentinel_paths, landsat_paths)
    """
    df = load_decisions_csv(csv_path)
    paths_dict = extract_paths_by_status(df, status)
    paths_dict = verify_file_paths(paths_dict)
    
    return paths_dict['sentinel2'], paths_dict['landsat']


if __name__ == "__main__":
    print("Example usage:")
    print("""
    # Show CSV summary
    show_csv_summary('label_decisions.csv')
    
    # Create train/val dataloaders
    train_loader = create_train_dataloader(
        'label_decisions.csv',
        labels_dir='path/to/labels',
        batch_size=8,
        target_size=(512, 512),
        normalize_per_image=True,
        clip_outliers=True
    )
    
    val_loader = create_val_dataloader(
        'label_decisions.csv',
        labels_dir='path/to/labels',
        batch_size=8,
        target_size=(512, 512),
        normalize_per_image=True,
        clip_outliers=True
    )
    
    # Or get file lists manually
    sentinel_paths, landsat_paths = get_file_lists_for_manual_dataloader(
        'label_decisions.csv', 'train'
    )
    """)