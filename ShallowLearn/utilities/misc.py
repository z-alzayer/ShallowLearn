"""
Miscellaneous utility functions.
Refactored from Misc.py to follow module organization.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
    """
    Flatten a nested dictionary.
    
    Parameters:
    -----------
    d : Dict[str, Any]
        Dictionary to flatten
    parent_key : str
        Parent key prefix
    sep : str
        Separator for nested keys
        
    Returns:
    --------
    Dict[str, Any]
        Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def safe_divide(a: Union[float, np.ndarray], b: Union[float, np.ndarray], 
                default: float = 0.0) -> Union[float, np.ndarray]:
    """
    Perform safe division avoiding division by zero.
    
    Parameters:
    -----------
    a : Union[float, np.ndarray]
        Numerator
    b : Union[float, np.ndarray]
        Denominator
    default : float
        Value to return when denominator is zero
        
    Returns:
    --------
    Union[float, np.ndarray]
        Result of division or default value
    """
    if isinstance(b, np.ndarray):
        result = np.full_like(b, default, dtype=float)
        mask = b != 0
        result[mask] = a[mask] / b[mask] if isinstance(a, np.ndarray) else a / b[mask]
        return result
    else:
        return a / b if b != 0 else default


def normalize_array(arr: np.ndarray, method: str = 'minmax') -> np.ndarray:
    """
    Normalize array using different methods.
    
    Parameters:
    -----------
    arr : np.ndarray
        Array to normalize
    method : str
        Normalization method ('minmax', 'zscore', 'unit')
        
    Returns:
    --------
    np.ndarray
        Normalized array
    """
    if method == 'minmax':
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max == arr_min:
            return np.zeros_like(arr)
        return (arr - arr_min) / (arr_max - arr_min)
    
    elif method == 'zscore':
        mean, std = arr.mean(), arr.std()
        if std == 0:
            return np.zeros_like(arr)
        return (arr - mean) / std
    
    elif method == 'unit':
        norm = np.linalg.norm(arr)
        if norm == 0:
            return arr
        return arr / norm
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split list into chunks of specified size.
    
    Parameters:
    -----------
    lst : List[Any]
        List to chunk
    chunk_size : int
        Size of each chunk
        
    Returns:
    --------
    List[List[Any]]
        List of chunks
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def unique_with_counts(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get unique values and their counts.
    
    Parameters:
    -----------
    arr : np.ndarray
        Input array
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        Unique values and their counts
    """
    unique_vals, counts = np.unique(arr, return_counts=True)
    return unique_vals, counts


def memory_usage(obj: Any) -> str:
    """
    Get memory usage of an object in human-readable format.
    
    Parameters:
    -----------
    obj : Any
        Object to check
        
    Returns:
    --------
    str
        Memory usage string
    """
    try:
        import sys
        size = sys.getsizeof(obj)
        
        # Add size of contents for containers
        if hasattr(obj, '__dict__'):
            size += sys.getsizeof(obj.__dict__)
        
        if hasattr(obj, '__len__'):
            if isinstance(obj, (list, tuple)):
                size += sum(sys.getsizeof(item) for item in obj)
            elif isinstance(obj, dict):
                size += sum(sys.getsizeof(k) + sys.getsizeof(v) for k, v in obj.items())
        
        # Convert to human readable
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.2f} {unit}"
            size /= 1024
        return f"{size:.2f} TB"
    
    except Exception:
        return "Unknown"


def validate_array_shape(arr: np.ndarray, expected_shape: Tuple[int, ...], 
                        name: str = "array") -> None:
    """
    Validate array shape matches expected shape.
    
    Parameters:
    -----------
    arr : np.ndarray
        Array to validate
    expected_shape : Tuple[int, ...]
        Expected shape
    name : str
        Name for error messages
        
    Raises:
    -------
    ValueError
        If shape doesn't match
    """
    if arr.shape != expected_shape:
        raise ValueError(f"{name} shape {arr.shape} doesn't match expected {expected_shape}")


def ensure_2d(arr: np.ndarray) -> np.ndarray:
    """
    Ensure array is 2D by adding dimensions if needed.
    
    Parameters:
    -----------
    arr : np.ndarray
        Input array
        
    Returns:
    --------
    np.ndarray
        2D array
    """
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    elif arr.ndim == 2:
        return arr
    else:
        raise ValueError(f"Cannot convert {arr.ndim}D array to 2D")