"""
Reef comparison visualization methods for L1C vs L2A analysis.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from skimage.transform import resize
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from .plot_utils import standardize_axes, create_dual_square_figure


def plot_l1c_l2a_comparison(l1c_results: Dict[str, Any], 
                           l2a_results: Dict[str, Any],
                           reef_name: str,
                           figsize_base: int = 6,
                           save_path: Optional[str] = None,
                           show_thumbnails: bool = True,
                           show_clusters: bool = True,
                           zoom: float = 0.15) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Create dual subplot comparison of L1C vs L2A processing for a single reef.
    
    Args:
        l1c_results: Results dict from L1C QuickLook processing
        l2a_results: Results dict from L2A QuickLook processing  
        reef_name: Name of the reef being analyzed
        figsize_base: Base size for square subplots
        save_path: Path to save SVG file
        show_thumbnails: Whether to overlay thumbnail images
        zoom: Zoom level for thumbnails
        
    Returns:
        Figure and axes objects
    """
    fig, (ax1, ax2) = create_dual_square_figure(figsize_base)
    
    # Colors for consistency across plots
    colors = ['#2ca02c', '#ff7f0e', '#d62728', '#1f77b4', '#9467bd', '#8c564b']
    
    # Plot L1C data
    _plot_single_analysis(ax1, l1c_results, 'L1C', colors, show_thumbnails, show_clusters, zoom)
    
    # Plot L2A data  
    _plot_single_analysis(ax2, l2a_results, 'L2A', colors, show_thumbnails, show_clusters, zoom)
    
    # Main title
    fig.suptitle(f'Reef Analysis: {reef_name}', fontsize=16, fontweight='bold', y=0.95)
    
    # Adjust layout to make room for legends outside plots
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.subplots_adjust(top=0.88)
    
    if save_path:
        plt.savefig(save_path, format='svg', bbox_inches='tight', dpi=300)
        print(f"✅ Saved: {save_path}")
    
    return fig, [ax1, ax2]


def _plot_single_analysis(ax: plt.Axes, 
                         results: Dict[str, Any], 
                         data_type: str,
                         colors: List[str],
                         show_thumbnails: bool = True,
                         show_clusters: bool = True,
                         zoom: float = 0.15) -> None:
    """
    Plot a single analysis (L1C or L2A) on given axes.
    
    Args:
        ax: Matplotlib axes to plot on
        results: QuickLook results dictionary
        data_type: 'L1C' or 'L2A'
        colors: Color palette
        show_thumbnails: Whether to show thumbnail images
        zoom: Thumbnail zoom level
    """
    transformed_data = results['transformed_data']
    labels = results['labels']
    processed_images = results.get('processed_images', [])
    reducer_name = results.get('reducer_name', 'PCA')
    
    if transformed_data is None or labels is None:
        ax.text(0.5, 0.5, f'No {data_type} data', transform=ax.transAxes, 
                ha='center', va='center', fontsize=14)
        ax.set_title(f'{data_type} Processing', fontsize=14, fontweight='bold')
        return
    
    # Get unique labels
    unique_labels = np.unique(labels)
    
    # Create scatter plots
    scatter_handles = []
    
    def plot_by_clusters():
        """Helper function for cluster-based plotting."""
        for i, label in enumerate(unique_labels):
            mask = labels == label
            color = colors[i % len(colors)]
            
            if label == -1:
                label_name = "Outliers"
            else:
                label_name = f"Cluster {label}"
                
            scatter = ax.scatter(
                transformed_data[mask, 0],
                transformed_data[mask, 1], 
                c=color,
                label=f'{label_name} ({sum(mask)})',
                s=60 if not show_thumbnails else 30,
                alpha=0.8 if not show_thumbnails else 0.6,
                edgecolors='white',
                linewidth=1,
                zorder=3
            )
            scatter_handles.append(scatter)
    
    def plot_no_clusters():
        """Plot all points without cluster labels."""
        scatter = ax.scatter(
            transformed_data[:, 0],
            transformed_data[:, 1], 
            c='#1f77b4',  # Single blue color
            label=None,  # No label for L1C
            s=60 if not show_thumbnails else 30,
            alpha=0.8 if not show_thumbnails else 0.6,
            edgecolors='white',
            linewidth=1,
            zorder=3
        )
        # Don't add to handles if no label
        if scatter.get_label():
            scatter_handles.append(scatter)
    
    # Determine plotting strategy
    if not show_clusters:
        # No clustering display
        if data_type == 'L2A' and 'metadata_df' in results and results['metadata_df'] is not None:
            metadata_df = results['metadata_df']
            if 'aot_retrieval_method' in metadata_df.columns:
                # For L2A without clusters, still show AOT methods
                aot_methods = metadata_df['aot_retrieval_method'].dropna().unique()
                
                for i, aot_method in enumerate(aot_methods):
                    if aot_method in ['N/A', 'Unknown', '']:
                        continue
                        
                    # Find indices of images with this AOT method
                    aot_mask = metadata_df['aot_retrieval_method'] == aot_method
                    aot_indices = metadata_df[aot_mask].index.tolist()
                    
                    if len(aot_indices) > 0 and len(aot_indices) <= len(transformed_data):
                        color = colors[i % len(colors)]
                        aot_clean = str(aot_method).replace('_', '-').replace('SEN2COR-', '')
                        
                        scatter = ax.scatter(
                            transformed_data[aot_indices, 0],
                            transformed_data[aot_indices, 1], 
                            c=color,
                            label=f'AOT-{aot_clean} ({len(aot_indices)})',
                            s=60 if not show_thumbnails else 30,
                            alpha=0.8 if not show_thumbnails else 0.6,
                            edgecolors='white',
                            linewidth=1,
                            zorder=3
                        )
                        scatter_handles.append(scatter)
            else:
                plot_no_clusters()
        else:
            # L1C without clusters - just plot points
            plot_no_clusters()
    else:
        # With clustering display (existing logic)
        if data_type == 'L2A' and 'metadata_df' in results and results['metadata_df'] is not None:
            metadata_df = results['metadata_df']
            if 'aot_retrieval_method' in metadata_df.columns and show_clusters:
                # Group by AOT retrieval method for L2A data
                aot_methods = metadata_df['aot_retrieval_method'].dropna().unique()
                
                for i, aot_method in enumerate(aot_methods):
                    if aot_method in ['N/A', 'Unknown', '']:
                        continue
                        
                    # Find indices of images with this AOT method
                    aot_mask = metadata_df['aot_retrieval_method'] == aot_method
                    aot_indices = metadata_df[aot_mask].index.tolist()
                    
                    if len(aot_indices) > 0 and len(aot_indices) <= len(transformed_data):
                        color = colors[i % len(colors)]
                        aot_clean = str(aot_method).replace('_', '-').replace('SEN2COR-', '')
                        
                        scatter = ax.scatter(
                            transformed_data[aot_indices, 0],
                            transformed_data[aot_indices, 1], 
                            c=color,
                            label=f'AOT-{aot_clean} ({len(aot_indices)})',
                            s=60 if not show_thumbnails else 30,
                            alpha=0.8 if not show_thumbnails else 0.6,
                            edgecolors='white',
                            linewidth=1,
                            zorder=3
                        )
                        scatter_handles.append(scatter)
            else:
                # Fallback to cluster-based plotting if no AOT data
                plot_by_clusters()
        else:
            # For L1C data, use cluster-based plotting
            plot_by_clusters()
    
    # Add thumbnails if available and requested
    if show_thumbnails and processed_images:
        # Sample thumbnails to avoid overcrowding
        n_images = min(len(processed_images), 12)
        indices = np.random.choice(len(processed_images), n_images, replace=False)
        
        for i in indices:
            try:
                thumbnail = processed_images[i]
                
                # Resize thumbnail
                resized_img = resize(
                    thumbnail,
                    output_shape=(int(thumbnail.shape[0] * zoom), 
                                int(thumbnail.shape[1] * zoom)),
                    anti_aliasing=True,
                    preserve_range=True,
                ).astype(thumbnail.dtype)
                
                # Get cluster color for border
                label = labels[i] 
                label_idx = list(unique_labels).index(label)
                border_color = colors[label_idx % len(colors)]
                
                imagebox = OffsetImage(resized_img, zoom=zoom)
                ab = AnnotationBbox(
                    imagebox,
                    (transformed_data[i, 0], transformed_data[i, 1]),
                    frameon=True,
                    bboxprops=dict(
                        edgecolor=border_color,
                        linewidth=1.5,
                        facecolor='white',
                        alpha=0.9
                    ),
                    zorder=4
                )
                ax.add_artist(ab)
                
            except Exception as e:
                print(f"Error adding thumbnail {i}: {e}")
    
    # Set axis limits with padding
    if len(transformed_data) > 0:
        x_min, x_max = transformed_data[:, 0].min(), transformed_data[:, 0].max()
        y_min, y_max = transformed_data[:, 1].min(), transformed_data[:, 1].max()
        
        x_range = x_max - x_min
        y_range = y_max - y_min
        padding_x = x_range * 0.15
        padding_y = y_range * 0.15
        
        ax.set_xlim(x_min - padding_x, x_max + padding_x)
        ax.set_ylim(y_min - padding_y, y_max + padding_y)
    
    # Labels and styling
    method_name = reducer_name.split('(')[0]  # Clean method name
    ax.set_xlabel(f'{method_name} Component 1', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{method_name} Component 2', fontsize=12, fontweight='bold') 
    ax.set_title(f'{data_type} Processing', fontsize=14, fontweight='bold')
    
    # Legend - place outside plot area
    if scatter_handles:
        legend = ax.legend(
            handles=scatter_handles,
            loc='center left',
            bbox_to_anchor=(1.02, 0.5),
            fontsize=9,
            frameon=True,
            fancybox=True,
            shadow=True,
            framealpha=0.9
        )
        legend.get_frame().set_linewidth(1)
    
    # Additional styling beyond standardize_axes
    pass  # standardize_axes already handles grid and tick params


def plot_reef_summary(reef_results: Dict[str, Dict], 
                     figsize: Tuple[int, int] = (20, 12),
                     save_path: Optional[str] = None) -> Tuple[plt.Figure, np.ndarray]:
    """
    Create summary plot showing all processed reefs.
    
    Args:
        reef_results: Dict mapping reef names to their L1C/L2A results
        figsize: Figure size
        save_path: Path to save summary plot
        
    Returns:
        Figure and axes array
    """
    n_reefs = len(reef_results)
    if n_reefs == 0:
        return None, None
        
    # Create grid layout
    n_cols = min(3, n_reefs)  # Max 3 columns
    n_rows = (n_reefs + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_reefs == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    colors = ['#2ca02c', '#ff7f0e', '#d62728', '#1f77b4', '#9467bd', '#8c564b']
    
    for i, (reef_name, reef_data) in enumerate(reef_results.items()):
        ax = axes[i]
        
        # Combine L1C and L2A data for this reef if available
        l1c_data = reef_data.get('L1C')
        l2a_data = reef_data.get('L2A')
        
        if l1c_data and l2a_data:
            # Plot both with different markers
            _plot_combined_reef_data(ax, l1c_data, l2a_data, reef_name, colors)
        elif l1c_data:
            _plot_single_analysis(ax, l1c_data, 'L1C Only', colors, show_thumbnails=False)
        elif l2a_data:
            _plot_single_analysis(ax, l2a_data, 'L2A Only', colors, show_thumbnails=False) 
        else:
            ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes,
                   ha='center', va='center', fontsize=12)
        
        ax.set_title(f'{reef_name}', fontsize=12, fontweight='bold')
    
    # Hide unused subplots
    for j in range(n_reefs, len(axes)):
        axes[j].set_visible(False)
    
    fig.suptitle('Reef Analysis Summary', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    if save_path:
        plt.savefig(save_path, format='svg', bbox_inches='tight', dpi=300)
        print(f"✅ Saved summary: {save_path}")
    
    return fig, axes


def _get_cluster_label(label: int, results: Dict[str, Any], data_type: str) -> str:
    """
    Get intelligent cluster label based on metadata.
    
    Args:
        label: Cluster label number
        results: QuickLook results dictionary
        data_type: 'L1C' or 'L2A'
        
    Returns:
        Descriptive cluster label
    """
    # For L2A data, always try to use AOT retrieval method regardless of clustering
    if data_type == 'L2A' and 'metadata_df' in results and results['metadata_df'] is not None:
        metadata_df = results['metadata_df']
        
        # Check if we have AOT retrieval method data
        if 'aot_retrieval_method' in metadata_df.columns:
            # Get all AOT methods (ignore cluster assignments for labeling)
            aot_methods = metadata_df['aot_retrieval_method'].dropna()
            
            if len(aot_methods) > 0:
                most_common_aot = aot_methods.value_counts().index[0]
                if most_common_aot not in ['N/A', 'Unknown', '']:
                    # Clean up the AOT method name for display
                    aot_clean = str(most_common_aot).replace('_', '-').replace('SEN2COR-', '')
                    return f"AOT-{aot_clean}"
    
    # Default cluster naming
    return f"Cluster {label}"


def _plot_combined_reef_data(ax: plt.Axes,
                           l1c_data: Dict,
                           l2a_data: Dict, 
                           reef_name: str,
                           colors: List[str]) -> None:
    """Plot L1C and L2A data together with different markers."""
    
    # Plot L1C data with circles
    l1c_transformed = l1c_data['transformed_data']
    l1c_labels = l1c_data['labels']
    
    if l1c_transformed is not None and l1c_labels is not None:
        unique_l1c = np.unique(l1c_labels)
        for i, label in enumerate(unique_l1c):
            mask = l1c_labels == label
            ax.scatter(
                l1c_transformed[mask, 0],
                l1c_transformed[mask, 1],
                c=colors[i % len(colors)],
                marker='o',
                s=50,
                alpha=0.7,
                edgecolors='white',
                linewidth=1,
                label=f'L1C-{label}' if label != -1 else 'L1C-Outliers'
            )
    
    # Plot L2A data with squares  
    l2a_transformed = l2a_data['transformed_data']
    l2a_labels = l2a_data['labels']
    
    if l2a_transformed is not None and l2a_labels is not None:
        unique_l2a = np.unique(l2a_labels)
        for i, label in enumerate(unique_l2a):
            mask = l2a_labels == label
            ax.scatter(
                l2a_transformed[mask, 0],
                l2a_transformed[mask, 1],
                c=colors[i % len(colors)],
                marker='s',
                s=50,
                alpha=0.7,
                edgecolors='white', 
                linewidth=1,
                label=f'L2A-{label}' if label != -1 else 'L2A-Outliers'
            )
    
    # Styling
    method_name = l1c_data.get('reducer_name', 'PCA').split('(')[0]
    ax.set_xlabel(f'{method_name} Component 1', fontsize=10)
    ax.set_ylabel(f'{method_name} Component 2', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=9)
    
    # Legend with small font
    ax.legend(fontsize=8, loc='best', frameon=True, framealpha=0.8)