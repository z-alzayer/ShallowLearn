"""
Density plotting utilities for visualization.
Refactored from DensityPlot.py to follow module organization.
"""

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.interpolate import interpn

# Optional import for scatter density
try:
    import mpl_scatter_density  # adds projection='scatter_density'
    from mpl_scatter_density import ScatterDensityArtist
    HAS_SCATTER_DENSITY = True
except ImportError:
    HAS_SCATTER_DENSITY = False
    mpl_scatter_density = None
    ScatterDensityArtist = None


# Custom colormap
white_viridis = LinearSegmentedColormap.from_list(
    'white_viridis', 
    [
        (0, '#ffffff'),
        (1e-20, '#440053'),
        (0.2, '#404388'),
        (0.4, '#2a788e'),
        (0.6, '#21a784'),
        (0.8, '#78d151'),
        (1, '#fde624'),
    ], 
    N=256
)


def set_limits(
    ax: plt.Axes, 
    x_scale: Optional[Tuple[float, float]] = None, 
    y_scale: Optional[Tuple[float, float]] = None
) -> None:
    """
    Set axis limits for a plot.
    
    Parameters:
    -----------
    ax : plt.Axes
        Matplotlib axes object
    x_scale : Optional[Tuple[float, float]]
        X-axis limits (min, max)
    y_scale : Optional[Tuple[float, float]]
        Y-axis limits (min, max)
    """
    if x_scale is not None:
        ax.set_xlim(x_scale)
    if y_scale is not None:
        ax.set_ylim(y_scale)


def density_scatter(
    x: np.ndarray,
    y: np.ndarray,
    ax: plt.Axes,
    sort: bool = True,
    bins: int = 20,
    **kwargs
) -> plt.Axes:
    """
    Create scatter plot colored by 2D histogram density.
    
    Parameters:
    -----------
    x : np.ndarray
        X coordinates
    y : np.ndarray
        Y coordinates
    ax : plt.Axes
        Matplotlib axes object
    sort : bool
        If True, sort points by density (densest plotted last)
    bins : int
        Number of bins for 2D histogram
    **kwargs
        Additional arguments for scatter plot
        
    Returns:
    --------
    plt.Axes
        Modified axes object
    """
    fig = ax.figure
    
    # Create 2D histogram
    data, x_e, y_e = np.histogram2d(x, y, bins=bins, density=True)
    
    # Interpolate density values at data points
    z = interpn(
        (0.5*(x_e[1:] + x_e[:-1]), 0.5*(y_e[1:]+y_e[:-1])),
        data,
        np.vstack([x, y]).T,
        method="splinef2d",
        bounds_error=False
    )

    # Replace NaN values with 0
    z[np.where(np.isnan(z))] = 0.0

    # Sort points by density if requested
    if sort:
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    # Create scatter plot
    ax.scatter(x, y, c=z, **kwargs)

    # Add colorbar
    norm = Normalize(vmin=np.min(z), vmax=np.max(z))
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm), ax=ax)
    cbar.ax.set_ylabel('Density')

    return ax


def kde_plot(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    x_scale: Optional[Tuple[float, float]] = None,
    y_scale: Optional[Tuple[float, float]] = None,
    cmap: str = 'coolwarm'
) -> None:
    """
    Create kernel density estimate plot using seaborn.
    
    Parameters:
    -----------
    ax : plt.Axes
        Matplotlib axes object
    x : np.ndarray
        X coordinates
    y : np.ndarray
        Y coordinates
    x_scale : Optional[Tuple[float, float]]
        X-axis limits
    y_scale : Optional[Tuple[float, float]]
        Y-axis limits
    cmap : str
        Colormap name
    """
    # Create KDE plot
    sns.kdeplot(x=x, y=y, ax=ax, cmap=cmap, fill=True)
    set_limits(ax, x_scale, y_scale)


def hist_2d_plot(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    x_scale: Optional[Tuple[float, float]] = None,
    y_scale: Optional[Tuple[float, float]] = None,
    bins: int = 30,
    cmap = None
) -> None:
    """
    Create 2D histogram plot.
    
    Parameters:
    -----------
    ax : plt.Axes
        Matplotlib axes object
    x : np.ndarray
        X coordinates
    y : np.ndarray
        Y coordinates
    x_scale : Optional[Tuple[float, float]]
        X-axis limits
    y_scale : Optional[Tuple[float, float]]
        Y-axis limits
    bins : int
        Number of bins for histogram
    cmap : optional
        Colormap (defaults to white_viridis)
    """
    if cmap is None:
        cmap = white_viridis
        
    # Create 2D histogram
    h = ax.hist2d(x, y, bins=bins, cmap=cmap)
    ax.figure.colorbar(h[3], ax=ax, label='Counts in bin')
    set_limits(ax, x_scale, y_scale)