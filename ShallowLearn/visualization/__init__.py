"""
Visualization utilities for ShallowLearn.
Clean plotting and display functions for remote sensing data.
"""

from .display import (
    # Core visualization functions
    create_rgb_image,
    plot_rgb,
    plot_discrete_image,
    plot_histogram,
    
    # Enhanced functions (less hardcoded)
    plot_rgb_enhanced,
    plot_color_space,
    
    # Color space functions (backwards compatibility)
    plot_hsv,
    plot_lab,
    plot_ycbcr,
    
    # Utility functions
    add_north_arrow_to_axis,
    plot_with_legend,
)

from .quicklook_viz import QuickLookVisualizer

__all__ = [
    # Core functions
    'create_rgb_image',
    'plot_rgb',
    'plot_discrete_image', 
    'plot_histogram',
    
    # Enhanced functions
    'plot_rgb_enhanced',
    'plot_color_space',
    
    # Color space functions
    'plot_hsv',
    'plot_lab', 
    'plot_ycbcr',
    
    # Utilities
    'add_north_arrow_to_axis',
    'plot_with_legend',
    
    # QuickLook visualization
    'QuickLookVisualizer',
]