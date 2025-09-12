"""
Machine Learning module for ShallowLearn
Contains dimensionality reduction, clustering, and analysis components
"""

from .quicklook_ml import (
    QuickLookConfig,
    QuickLookFilter,
    DimensionalityReducer,
    PCAReducer,
    TSNEReducer,
    UMAPReducer,
    SVDReducer,
    ThumbnailLoader
)

__all__ = [
    'QuickLookConfig',
    'QuickLookFilter',
    'DimensionalityReducer',
    'PCAReducer',
    'TSNEReducer',
    'UMAPReducer',
    'SVDReducer',
    'ThumbnailLoader'
]