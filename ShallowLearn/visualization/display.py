"""
Visualization utilities for remote sensing data.
Clean plotting functions with minimal dependencies.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, to_rgba
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from sklearn.preprocessing import minmax_scale
from skimage.color import rgb2lab, rgb2hsv, rgb2ycbcr
from typing import List, Dict, Optional, Tuple, Union
import warnings

# Suppress matplotlib warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')


def create_rgb_image(img: np.ndarray, 
                    band_indices: List[int], 
                    stretch: bool = True) -> np.ndarray:
    """
    Creates an RGB image from multispectral data using specified band indices.

    Parameters:
    -----------
    img : np.ndarray
        Input image array with shape (height, width, bands)
    band_indices : List[int]
        List of 3 band indices for R, G, B channels
    stretch : bool, default=True
        Whether to apply min-max stretch to each channel

    Returns:
    --------
    np.ndarray
        RGB image array with shape (height, width, 3) and dtype uint8
    """
    if len(band_indices) != 3:
        raise ValueError("Exactly 3 band indices required for RGB")
    
    img_shape = img.shape
    rgb_channels = []
    
    for band_idx in band_indices:
        if band_idx >= img_shape[2]:
            raise ValueError(f"Band index {band_idx} out of bounds for image with {img_shape[2]} bands")
        
        channel = img[:, :, band_idx].astype(float)
        
        if stretch:
            # Apply min-max stretch
            channel_flat = channel.flatten()
            channel_stretched = minmax_scale(channel_flat, feature_range=(0, 255), copy=True)
            channel = channel_stretched.reshape(img_shape[0], img_shape[1])
        else:
            # Simple scaling to 0-255 range
            channel = np.clip(channel * 255 / np.max(channel), 0, 255)
        
        rgb_channels.append(channel.astype(np.uint8))
    
    return np.dstack(rgb_channels)


def plot_rgb(img: np.ndarray, 
            band_indices: List[int],
            title: str = "RGB Image",
            figsize: Tuple[int, int] = (8, 8),
            show: bool = True) -> Optional[plt.Figure]:
    """
    Plots an RGB image using specified band indices.

    Parameters:
    -----------
    img : np.ndarray
        Input image array with shape (height, width, bands)
    band_indices : List[int]
        List of 3 band indices for R, G, B channels
    title : str, default="RGB Image"
        Title for the plot
    figsize : Tuple[int, int], default=(8, 8)
        Figure size
    show : bool, default=True
        Whether to display the plot

    Returns:
    --------
    plt.Figure or None
        Figure object if show=False, otherwise None
    """
    rgb_image = create_rgb_image(img, band_indices)
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(rgb_image)
    ax.set_title(title)
    ax.axis('off')
    
    if show:
        plt.show()
        return None
    return fig


def plot_color_space(img: np.ndarray, 
                    band_indices: List[int],
                    color_space: str = 'hsv',
                    channel: int = 0,
                    title: Optional[str] = None,
                    figsize: Tuple[int, int] = (8, 8),
                    show: bool = True) -> Optional[plt.Figure]:
    """
    Converts RGB image to different color spaces and plots a specific channel.

    Parameters:
    -----------
    img : np.ndarray
        Input image array
    band_indices : List[int]
        List of 3 band indices for R, G, B channels
    color_space : str, default='hsv'
        Color space to convert to ('hsv', 'lab', 'ycbcr')
    channel : int, default=0
        Channel to display (0, 1, or 2)
    title : str, optional
        Title for the plot
    figsize : Tuple[int, int], default=(8, 8)
        Figure size
    show : bool, default=True
        Whether to display the plot

    Returns:
    --------
    plt.Figure or None
        Figure object if show=False, otherwise None
    """
    rgb_image = create_rgb_image(img, band_indices)
    
    # Convert to specified color space
    if color_space.lower() == 'hsv':
        converted = rgb2hsv(rgb_image)
        channel_names = ['Hue', 'Saturation', 'Value']
        cmap = 'hsv' if channel == 0 else 'gray'
    elif color_space.lower() == 'lab':
        converted = rgb2lab(rgb_image)
        channel_names = ['Lightness', 'a*', 'b*']
        cmap = 'gray'
    elif color_space.lower() == 'ycbcr':
        converted = rgb2ycbcr(rgb_image)
        channel_names = ['Y (Luma)', 'Cb (Blue-diff)', 'Cr (Red-diff)']
        cmap = 'gray'
    else:
        raise ValueError(f"Unsupported color space: {color_space}")
    
    if channel >= converted.shape[2]:
        raise ValueError(f"Channel {channel} out of bounds for {color_space.upper()} color space")
    
    channel_data = converted[:, :, channel]
    
    if title is None:
        title = f"{color_space.upper()} - {channel_names[channel]} Channel"
    
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(channel_data, cmap=cmap)
    ax.set_title(title)
    ax.axis('off')
    plt.colorbar(im, ax=ax)
    
    if show:
        plt.show()
        return None
    return fig


def plot_histogram(img: np.ndarray, 
                  channels: Optional[List[int]] = None,
                  bins: int = 50, 
                  min_value: float = 1,
                  channel_names: Optional[List[str]] = None,
                  title: str = "Histogram",
                  figsize: Tuple[int, int] = (10, 6),
                  show: bool = True) -> Optional[plt.Figure]:
    """
    Plots histograms for specified channels.

    Parameters:
    -----------
    img : np.ndarray
        Input image array
    channels : List[int], optional
        List of channel indices to plot. If None, plots all channels
    bins : int, default=50
        Number of bins for histogram
    min_value : float, default=1
        Minimum value threshold for filtering
    channel_names : List[str], optional
        Names for channels in legend
    title : str, default="Histogram"
        Title for the plot
    figsize : Tuple[int, int], default=(10, 6)
        Figure size
    show : bool, default=True
        Whether to display the plot

    Returns:
    --------
    plt.Figure or None
        Figure object if show=False, otherwise None
    """
    if len(img.shape) == 2:
        # Single channel image
        img = img[:, :, np.newaxis]
    
    if channels is None:
        channels = list(range(img.shape[2]))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.linspace(0, np.max(img), bins)
    
    for i, channel_idx in enumerate(channels):
        if channel_idx >= img.shape[2]:
            continue
            
        channel_data = img[:, :, channel_idx].flatten()
        channel_data = channel_data[channel_data >= min_value]
        
        if len(channel_data) == 0:
            continue
            
        histogram, _ = np.histogram(channel_data, bins=bins, range=(0, np.max(img)))
        
        # Use channel names if provided
        if channel_names and i < len(channel_names):
            label = channel_names[i]
        else:
            label = f'Channel {channel_idx + 1}'
        
        ax.plot(x, histogram, label=label, alpha=0.7)
    
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if show:
        plt.show()
        return None
    return fig


def plot_discrete_image(arr: np.ndarray, 
                       value_labels: Optional[Dict] = None,
                       colors: Optional[List] = None,
                       pixel_scale: float = 10,
                       title: str = "Discrete Image",
                       figsize: Tuple[int, int] = (10, 8),
                       show: bool = True) -> Optional[plt.Figure]:
    """
    Plots a discrete array with custom colors and labels.

    Parameters:
    -----------
    arr : np.ndarray
        Input discrete array
    value_labels : Dict, optional
        Dictionary mapping values to labels
    colors : List, optional
        List of colors for each unique value
    pixel_scale : float, default=10
        Scale for the scale bar (pixels per km)
    title : str, default="Discrete Image"
        Title for the plot
    figsize : Tuple[int, int], default=(10, 8)
        Figure size
    show : bool, default=True
        Whether to display the plot

    Returns:
    --------
    plt.Figure or None
        Figure object if show=False, otherwise None
    """
    if len(arr.shape) == 1:
        arr = arr.reshape(-1, 1)
    
    unique_labels = np.unique(arr)
    num_labels = len(unique_labels)
    
    # Create label to integer mapping
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    int_arr = np.vectorize(label_to_int.get)(arr)
    
    # Create colormap
    if colors is None:
        colors = plt.get_cmap('viridis')(np.linspace(0, 1, num_labels))
    elif len(colors) < num_labels:
        # Extend colors if not enough provided
        base_colors = plt.get_cmap('viridis')(np.linspace(0, 1, num_labels))
        for i, color in enumerate(colors):
            base_colors[i] = to_rgba(color)
        colors = base_colors
    else:
        colors = [to_rgba(c) for c in colors[:num_labels]]
    
    cmap = ListedColormap(colors)
    
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(int_arr, cmap=cmap)
    
    # Create colorbar
    cbar = fig.colorbar(im, ticks=np.arange(num_labels), drawedges=True)
    cbar.set_label('Labels')
    
    # Set tick labels
    if value_labels:
        tick_labels = [value_labels.get(label, str(label)) for label in unique_labels]
    else:
        tick_labels = [str(label) for label in unique_labels]
    cbar.set_ticklabels(tick_labels)
    
    # Add scale bar
    scalebar = AnchoredSizeBar(ax.transData,
                              10 * pixel_scale, '1 km', 'lower right',
                              pad=0.25,
                              color='white',
                              frameon=False,
                              size_vertical=1)
    ax.add_artist(scalebar)
    
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    if show:
        plt.show()
        return None
    return fig


def plot_with_legend(array: np.ndarray, 
                    value_dict: Dict,
                    title: str = "Classified Image",
                    figsize: Tuple[int, int] = (10, 8),
                    show: bool = True) -> Optional[plt.Figure]:
    """
    Plots a 2D array with a legend using distinct colors for discrete class labels.

    Parameters:
    -----------
    array : np.ndarray
        2D array to be plotted
    value_dict : Dict
        Dictionary mapping values in the array to labels
    title : str, default="Classified Image"
        Title for the plot
    figsize : Tuple[int, int], default=(10, 8)
        Figure size
    show : bool, default=True
        Whether to display the plot

    Returns:
    --------
    plt.Figure or None
        Figure object if show=False, otherwise None
    """
    n_classes = len(value_dict)
    cmap = plt.cm.get_cmap('Set3', n_classes)
    
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(array, cmap=cmap)
    
    # Create color map index for each discrete value
    colors = [cmap(i) for i in range(n_classes)]
    
    # Create legend patches
    patches = [mpatches.Patch(color=colors[i], label=label) 
              for i, (value, label) in enumerate(value_dict.items())]
    
    # Add legend
    ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    ax.set_title(title)
    
    if show:
        plt.show()
        return None
    return fig


def plot_geotiff_with_coordinates(image_data: np.ndarray, 
                                bounds: Tuple[float, float, float, float],
                                title: str = "Map with coordinates",
                                figsize: Tuple[int, int] = (10, 10),
                                add_scalebar: bool = True,
                                add_north_arrow: bool = True,
                                show: bool = True) -> Optional[plt.Figure]:
    """
    Plots GeoTIFF with UTM coordinates, scale bar, and north arrow.

    Parameters:
    -----------
    image_data : np.ndarray
        2D or 3D array containing the raster data
    bounds : Tuple[float, float, float, float]
        Bounding coordinates (left, bottom, right, top)
    title : str, default="Map with coordinates"
        Title for the plot
    figsize : Tuple[int, int], default=(10, 10)
        Figure size
    add_scalebar : bool, default=True
        Whether to add a scale bar
    add_north_arrow : bool, default=True
        Whether to add a north arrow
    show : bool, default=True
        Whether to display the plot

    Returns:
    --------
    plt.Figure or None
        Figure object if show=False, otherwise None
    """
    from matplotlib.ticker import ScalarFormatter
    try:
        from matplotlib_scalebar.scalebar import ScaleBar
        scalebar_available = True
    except ImportError:
        scalebar_available = False
        if add_scalebar:
            print("Warning: matplotlib_scalebar not available. Scalebar will not be added.")
    
    left, bottom, right, top = bounds
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Handle NaN values for RGB images
    if image_data.ndim == 3 and image_data.shape[2] == 3 and np.isnan(image_data).any():
        alpha = np.ones_like(image_data[..., 0])
        alpha[np.isnan(image_data).any(axis=-1)] = 0
        image_data = np.dstack((image_data, alpha))
    
    ax.imshow(image_data, extent=[left, right, bottom, top])
    
    # Format coordinates without scientific notation
    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    
    # Add scale bar
    if add_scalebar and scalebar_available:
        scalebar = ScaleBar(1, location='lower right', scale_loc='bottom', box_alpha=0.7)
        ax.add_artist(scalebar)
    
    # Add north arrow
    if add_north_arrow:
        add_north_arrow_to_axis(ax)
    
    ax.set_title(title)
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    
    if show:
        plt.show()
        return None
    return fig


def add_north_arrow_to_axis(ax: plt.Axes, 
                           relative_position: Tuple[float, float] = (0.05, 0.05),
                           arrow_length: float = 0.05,
                           text_offset: float = -0.02) -> None:
    """
    Adds a north arrow to an axis.

    Parameters:
    -----------
    ax : plt.Axes
        Matplotlib axis object
    relative_position : Tuple[float, float], default=(0.05, 0.05)
        Relative position of the arrow (0-1 range)
    arrow_length : float, default=0.05
        Length of the arrow relative to axis size
    text_offset : float, default=-0.02
        Text offset relative to axis size
    """
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    
    x = xlim[0] + (xlim[1] - xlim[0]) * relative_position[0]
    y = ylim[0] + (ylim[1] - ylim[0]) * relative_position[1]
    
    ax.arrow(x, y, 0, arrow_length * (ylim[1] - ylim[0]), 
            head_width=0.02 * (xlim[1] - xlim[0]), 
            head_length=0.03 * (ylim[1] - ylim[0]), 
            fc='black', ec='black')
    ax.text(x, y + text_offset * (ylim[1] - ylim[0]), 'N', 
           horizontalalignment='center', verticalalignment='center', 
           fontsize=12, fontweight='bold', color='black')


def create_animation_frames(images: List[np.ndarray],
                          band_indices: List[int],
                          titles: Optional[List[str]] = None,
                          figsize: Tuple[int, int] = (8, 8)) -> List[plt.Figure]:
    """
    Creates a list of figure frames for animation.

    Parameters:
    -----------
    images : List[np.ndarray]
        List of image arrays
    band_indices : List[int]
        Band indices for RGB visualization
    titles : List[str], optional
        List of titles for each frame
    figsize : Tuple[int, int], default=(8, 8)
        Figure size

    Returns:
    --------
    List[plt.Figure]
        List of figure objects for animation
    """
    frames = []
    
    for i, img in enumerate(images):
        title = titles[i] if titles and i < len(titles) else f"Frame {i+1}"
        fig = plot_rgb(img, band_indices, title=title, figsize=figsize, show=False)
        frames.append(fig)
    
    return frames


def plot_rgb_enhanced(
    img: np.ndarray, 
    band_indices: Optional[List[int]] = None,
    band_mapping: Optional[Dict] = None,
    band_names: Optional[List[str]] = None,
    stretch: bool = True,
    plot: bool = False,
    title: str = "RGB Image",
    figsize: Tuple[int, int] = (10, 8)
) -> Union[np.ndarray, None]:
    """
    Enhanced RGB plotting function with flexible band selection and reduced hardcoding.
    
    This function replaces ImageHelper.plot_rgb with improved flexibility and reduced 
    dependency on hardcoded band mappings.
    
    Parameters:
    -----------
    img : np.ndarray
        Input image array with shape (height, width, bands)
    band_indices : List[int], optional
        List of 3 band indices for R, G, B channels. If None, defaults to [3, 2, 1] 
        (which corresponds to typical Red, Green, Blue for Sentinel-2)
    band_mapping : Dict, optional
        Band mapping dictionary for converting band names to indices
    band_names : List[str], optional  
        List of 3 band names (e.g., ['B04', 'B03', 'B02']) to use with band_mapping
    stretch : bool, default=True
        Whether to apply min-max stretch to enhance contrast
    plot : bool, default=False
        Whether to display the image plot using matplotlib
    title : str, default="RGB Image"
        Title for the plot
    figsize : Tuple[int, int], default=(10, 8)
        Figure size if plotting
        
    Returns:
    --------
    np.ndarray or None
        RGB image array with shape (height, width, 3) and dtype uint8 if plot=False,
        otherwise None
    """
    # Determine band indices
    if band_indices is None:
        if band_names and band_mapping:
            # Use band mapping to convert names to indices
            band_indices = [band_mapping[band]['index'] for band in band_names]
        elif band_names is None and band_mapping is None:
            # Default to typical RGB bands for Sentinel-2 (B04=Red, B03=Green, B02=Blue)
            band_indices = [3, 2, 1]  # Assuming 0-indexed bands
        else:
            raise ValueError("Either band_indices or both band_names and band_mapping must be provided")
    
    if len(band_indices) != 3:
        raise ValueError("Exactly 3 band indices required for RGB")
    
    # Validate band indices
    for idx in band_indices:
        if idx >= img.shape[2]:
            raise ValueError(f"Band index {idx} out of bounds for image with {img.shape[2]} bands")
    
    img_shape = img.shape
    rgb_channels = []
    
    # Extract and process each channel
    for band_idx in band_indices:
        channel = img[:, :, band_idx].astype(float)
        
        if stretch:
            # Apply min-max stretch
            channel = minmax_scale(
                channel.flatten(), 
                feature_range=(0, 255), 
                axis=0, 
                copy=True
            ).reshape(img_shape[0], img_shape[1])
        else:
            # Simple clipping to 0-255 range
            channel = np.clip(channel, 0, 255)
        
        rgb_channels.append(np.uint8(channel))
    
    # Stack channels to create RGB image
    rgb = np.dstack(rgb_channels)
    
    if plot:
        plt.figure(figsize=figsize)
        plt.imshow(rgb)
        plt.title(title)
        plt.axis('off')
        plt.show()
        return None
    
    return rgb


def plot_color_space(
    img: np.ndarray,
    color_space: str = 'hsv',
    band_indices: Optional[List[int]] = None,
    band_mapping: Optional[Dict] = None,
    band_names: Optional[List[str]] = None,
    plot: bool = False,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> Union[np.ndarray, None]:
    """
    Convert image to different color spaces with flexible band selection.
    
    This function replaces ImageHelper functions like plot_hsv, plot_lab, plot_ycbcr
    with a unified interface.
    
    Parameters:
    -----------
    img : np.ndarray
        Input image array with shape (height, width, bands)
    color_space : str, default='hsv'
        Target color space ('hsv', 'lab', 'ycbcr')
    band_indices : List[int], optional
        List of 3 band indices for R, G, B channels used in conversion
    band_mapping : Dict, optional
        Band mapping dictionary for converting band names to indices  
    band_names : List[str], optional
        List of 3 band names to use with band_mapping
    plot : bool, default=False
        Whether to display the converted image
    title : str, optional
        Title for the plot. If None, auto-generated based on color_space
    figsize : Tuple[int, int], default=(10, 8)
        Figure size if plotting
        
    Returns:
    --------
    np.ndarray or None
        Converted image array if plot=False, otherwise None
        
    Raises:
    -------
    ValueError
        If color_space is not supported
    """
    # First create RGB image
    rgb_img = plot_rgb_enhanced(
        img, 
        band_indices=band_indices,
        band_mapping=band_mapping,
        band_names=band_names,
        stretch=True,
        plot=False
    )
    
    # Convert to requested color space
    if color_space.lower() == 'hsv':
        converted_img = rgb2hsv(rgb_img)
        default_title = "HSV Color Space"
    elif color_space.lower() == 'lab':
        converted_img = rgb2lab(rgb_img)  
        default_title = "LAB Color Space"
    elif color_space.lower() == 'ycbcr':
        converted_img = rgb2ycbcr(rgb_img)
        default_title = "YCbCr Color Space"
    else:
        raise ValueError(f"Unsupported color space: {color_space}")
    
    if plot:
        if title is None:
            title = default_title
            
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        fig.suptitle(title)
        
        channel_names = {
            'hsv': ['Hue', 'Saturation', 'Value'],
            'lab': ['Lightness', 'A*', 'B*'],
            'ycbcr': ['Luma', 'Chroma Blue', 'Chroma Red']
        }
        
        names = channel_names.get(color_space.lower(), ['Channel 1', 'Channel 2', 'Channel 3'])
        
        for i in range(3):
            axes[i].imshow(converted_img[:, :, i], cmap='gray')
            axes[i].set_title(names[i])
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
        return None
    
    return converted_img


# Backwards compatibility aliases
plot_rgb = plot_rgb_enhanced  # Direct alias for backwards compatibility
plot_hsv = lambda img, plot=False: plot_color_space(img, 'hsv', plot=plot)
plot_lab = lambda img, plot=False: plot_color_space(img, 'lab', plot=plot) 
plot_ycbcr = lambda img, plot=False: plot_color_space(img, 'ycbcr', plot=plot)