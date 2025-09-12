"""
Visualization utilities for consistent plot formatting.
"""

import matplotlib.pyplot as plt


def standardize_axes(ax):
    """
    Standardize axes formatting with auto aspect ratio.
    
    Args:
        ax: Matplotlib axes object to standardize
    """
    ax.set_aspect('auto')  # Remove aspect ratio constraints
    ax.tick_params(axis='both', labelsize=12) 
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)


def create_square_figure(figsize_base=6):
    """
    Create a square figure for single plots.
    
    Args:
        figsize_base: Base size for the square figure
        
    Returns:
        fig, ax: Matplotlib figure and axes
    """
    fig, ax = plt.subplots(figsize=(figsize_base, figsize_base))
    standardize_axes(ax)
    return fig, ax


def create_dual_square_figure(figsize_base=6):
    """
    Create dual subplot figure with square subplots.
    
    Args:
        figsize_base: Base size for each subplot
        
    Returns:
        fig, (ax1, ax2): Matplotlib figure and dual axes
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(figsize_base*2, figsize_base))
    standardize_axes(ax1)
    standardize_axes(ax2)
    return fig, (ax1, ax2)