"""
QuickLook Visualization Module
Handles all visualization aspects of QuickLook analysis, separate from ML processing
"""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from skimage.transform import resize
from .plot_utils import standardize_axes, create_square_figure


class QuickLookVisualizer:
    """Handles visualization of QuickLook results - separated from ML processing"""

    def __init__(self, quicklook_processor):
        """Initialize with a QuickLookProcessor instance from ml module"""
        self.processor = quicklook_processor
        self.transformed_data = quicklook_processor.transformed_data
        self.labels = quicklook_processor.labels
        self.images = quicklook_processor.processed_images
        self.metadata_df = quicklook_processor.metadata_df

        # Generate class dictionary for visualization
        self.class_dict = self._generate_class_dict()

        # For compatibility with existing visualization code
        self.thumbnails = self.images  # processed images act as thumbnails
        self.products = self._create_product_objects()

    def _generate_class_dict(self):
        """Generate a class dictionary for visualization colors and labels."""
        if self.labels is None:
            return {}

        unique_labels = np.unique(self.labels)
        colors = [
            "#2ca02c",
            "#ff7f0e",
            "#d62728",
            "#1f77b4",
            "#9467bd",
            "#8c564b",
            "#e377c2",
        ]

        class_dict = {}
        for i, label in enumerate(unique_labels):
            color = colors[i % len(colors)]

            # Generate meaningful label names based on cluster characteristics
            if label == -1:
                name = "Outliers"
            else:
                # For L2A data, try to use AOT retrieval method for labeling
                name = self._get_cluster_name(label)

            class_dict[label] = (color, name)

        return class_dict
    
    def _get_cluster_name(self, label):
        """Generate cluster name based on metadata, especially AOT retrieval method for L2A."""
        if self.metadata_df is None:
            return f"Cluster {label}"
        
        # Get images belonging to this cluster
        cluster_mask = self.labels == label
        cluster_metadata = self.metadata_df[cluster_mask]
        
        if len(cluster_metadata) == 0:
            return f"Cluster {label}"
        
        # Check if this is L2A data by looking for AOT_RETRIEVAL_METHOD
        aot_methods = cluster_metadata['aot_retrieval_method'].dropna()
        
        if len(aot_methods) > 0:
            # Get most common AOT retrieval method in this cluster
            most_common_aot = aot_methods.value_counts().index[0]
            if most_common_aot != 'N/A':
                return f"AOT-{most_common_aot}"
        
        # Check processing level
        processing_levels = cluster_metadata['processing_level'].dropna()
        if len(processing_levels) > 0:
            most_common_level = processing_levels.value_counts().index[0]
            if 'L2A' in most_common_level or 'Level-2A' in most_common_level:
                return f"L2A-C{label}"
            elif 'L1C' in most_common_level or 'Level-1C' in most_common_level:
                return f"L1C-C{label}"
        
        return f"Cluster {label}"

    def _create_product_objects(self):
        """Create product-like objects for compatibility with existing viz code."""
        products = []

        if self.metadata_df is not None:
            for _, row in self.metadata_df.iterrows():
                # Create a simple product object with minimal required attributes
                product = type(
                    "Product",
                    (),
                    {
                        "product_id": Path(row["file_path"]).name,
                        "satellite": row.get("satellite_type", "unknown"),
                        "cloud_cover": row.get("cloud_cover", 0),
                        "acquisition_date": "2023-01-01",  # Default date if not available
                    },
                )()
                products.append(product)

        return products

    def plot_clusters_scatter(self, figsize_base=8, save_path=None):
        """Create scatter plot of clusters in reduced dimensional space"""
        if self.transformed_data is None:
            raise ValueError(
                "No transformed data available. Run process_products first."
            )

        fig, ax = create_square_figure(figsize_base)

        # Get colors and labels for legend
        unique_labels = np.unique(self.labels)
        colors = []
        legend_labels = []

        for label in unique_labels:
            color, name = self.class_dict.get(label, ("#808080", f"Cluster_{label}"))
            colors.append(color)
            legend_labels.append(f"{name} ({np.sum(self.labels == label)})")

        # Create custom colormap
        cmap = ListedColormap(colors)

        # Plot scatter
        scatter = ax.scatter(
            self.transformed_data[:, 0],
            self.transformed_data[:, 1],
            c=self.labels,
            cmap=cmap,
            s=50,
            alpha=0.7,
        )

        # Add colorbar with custom labels
        cbar = plt.colorbar(scatter, ticks=unique_labels)
        cbar.set_ticklabels(legend_labels)

        # Set labels
        method_name = self.processor.reducer.get_name()
        ax.set_xlabel(f"{method_name} Component 1", fontsize=12)
        ax.set_ylabel(f"{method_name} Component 2", fontsize=12)
        ax.set_title(f"QuickLook Clustering Results ({method_name})", fontsize=14)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig, ax

    def plot_thumbnails_on_scatter(
        self,
        show_points=True,
        show_thumbnails=True,
        zoom=0.1,
        figsize_base=10,
        max_images=50,
        sample_method="random",
        add_borders=True,
        save_path=None,
    ):
        """
        Plot thumbnail images on their cluster coordinates with flexible options

        Args:
            show_points: Whether to show scatter points underneath thumbnails
            show_thumbnails: Whether to show thumbnail images
            zoom: Zoom level for thumbnails (0.05-0.2 recommended)
            figsize_base: Base size for square figure
            max_images: Maximum number of thumbnails to show (None = all)
            sample_method: 'random', 'cluster' (sample from each cluster), or 'all'
            add_borders: Whether to add colored borders to thumbnails matching clusters
            save_path: Path to save the plot
        """
        if self.transformed_data is None:
            raise ValueError("No data available. Run process_products first.")

        fig, ax = create_square_figure(figsize_base)

        # Get unique labels and colors
        unique_labels = np.unique(self.labels)
        colors = ["#2ca02c", "#ff7f0e", "#d62728", "#1f77b4", "#808080"]

        # Plot scatter points if requested
        if show_points:
            for i, label in enumerate(unique_labels):
                mask = self.labels == label
                label_name = self.class_dict.get(label, (None, f"Cluster {label}"))[1]

                ax.scatter(
                    self.transformed_data[mask, 0],
                    self.transformed_data[mask, 1],
                    c=colors[i % len(colors)],
                    label=f"{label_name} ({sum(mask)})",
                    s=50 if not show_thumbnails else 20,
                    alpha=0.7 if not show_thumbnails else 0.3,
                    edgecolors="black" if not show_thumbnails else "none",
                    linewidth=0.5 if not show_thumbnails else 0,
                )

        # Add thumbnails if requested
        if show_thumbnails and self.thumbnails:
            # Determine which thumbnails to show
            if sample_method == "all" or max_images is None:
                indices = range(len(self.thumbnails))
            elif sample_method == "cluster":
                # Sample evenly from each cluster
                indices = []
                for label in unique_labels:
                    mask = self.labels == label
                    label_indices = np.where(mask)[0]
                    n_samples = min(
                        max_images // len(unique_labels), len(label_indices)
                    )
                    if n_samples > 0:
                        sampled = np.random.choice(
                            label_indices, n_samples, replace=False
                        )
                        indices.extend(sampled)
            else:  # random
                n_images = min(len(self.thumbnails), max_images)
                indices = np.random.choice(
                    len(self.thumbnails), n_images, replace=False
                )

            # Add thumbnails
            for i in indices:
                try:
                    thumbnail = self.thumbnails[i]
                    original_dtype = thumbnail.dtype

                    # Resize thumbnail
                    shape = thumbnail.shape
                    resized_img = resize(
                        thumbnail,
                        output_shape=(int(shape[0] * zoom), int(shape[1] * zoom)),
                        anti_aliasing=True,
                        preserve_range=True,
                    ).astype(original_dtype)

                    # Create image box
                    imagebox = OffsetImage(resized_img, zoom=zoom)

                    # Add border if requested
                    if add_borders:
                        label = self.labels[i]
                        label_idx = list(unique_labels).index(label)
                        border_color = colors[label_idx % len(colors)]
                        ab = AnnotationBbox(
                            imagebox,
                            (self.transformed_data[i, 0], self.transformed_data[i, 1]),
                            frameon=True,
                            bboxprops=dict(edgecolor=border_color, linewidth=2),
                        )
                    else:
                        ab = AnnotationBbox(
                            imagebox,
                            (self.transformed_data[i, 0], self.transformed_data[i, 1]),
                            frameon=False,
                        )

                    ax.add_artist(ab)
                except Exception as e:
                    print(f"Error adding thumbnail {i}: {e}")

        # Set axis limits with some padding
        x_min, x_max = (
            self.transformed_data[:, 0].min(),
            self.transformed_data[:, 0].max(),
        )
        y_min, y_max = (
            self.transformed_data[:, 1].min(),
            self.transformed_data[:, 1].max(),
        )

        x_padding = (x_max - x_min) * 0.1
        y_padding = (y_max - y_min) * 0.1

        ax.set_xlim(x_min - x_padding, x_max + x_padding)
        ax.set_ylim(y_min - y_padding, y_max + y_padding)

        # Labels and legend
        method_name = (
            self.processor.reducer.get_name()
            if hasattr(self.processor, "reducer")
            else "Reduced"
        )
        ax.set_xlabel(f"{method_name} Component 1", fontsize=12)
        ax.set_ylabel(f"{method_name} Component 2", fontsize=12)

        # Title based on options
        if show_thumbnails and self.thumbnails:
            n_shown = len(list(indices))
            title = f"Satellite Thumbnails in {method_name} Space ({n_shown}/{len(self.thumbnails)} shown)"
        else:
            title = f"Satellite Clustering in {method_name} Space"
        ax.set_title(title, fontsize=14)

        if show_points:
            ax.legend(loc="best")
        
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig, ax

    def plot_temporal_distribution(self, figsize=(14, 8), save_path=None):
        """Plot temporal distribution of products by cluster"""
        if not self.products:
            raise ValueError("No products available")

        # Extract dates from products
        dates = []
        cloud_covers = []

        for product in self.products:
            try:
                # Handle different date formats
                date_str = product.acquisition_date
                if "T" in date_str:  # ISO format
                    date = pd.to_datetime(date_str.split("T")[0])
                else:
                    date = pd.to_datetime(date_str.split()[0])  # Split by space
                dates.append(date)
                cloud_covers.append(product.cloud_cover)
            except Exception as e:
                print(f"Error parsing date for {product.product_id}: {e}")
                continue

        if not dates:
            print("No valid dates found in products")
            return None, None

        # Create DataFrame for easier plotting
        df = pd.DataFrame(
            {
                "date": dates,
                "cloud_cover": cloud_covers,
                "label": self.labels[
                    : len(dates)
                ],  # In case some dates failed to parse
            }
        )

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

        # Plot 1: Temporal distribution by cluster
        unique_labels = np.unique(self.labels)
        for label in unique_labels:
            cluster_data = df[df["label"] == label]
            if len(cluster_data) > 0:
                color, name = self.class_dict.get(
                    label, ("#808080", f"Cluster_{label}")
                )
                ax1.scatter(
                    cluster_data["date"],
                    cluster_data["cloud_cover"],
                    c=color,
                    label=f"{name} ({len(cluster_data)})",
                    alpha=0.7,
                    s=50,
                )

        ax1.set_xlabel("Date", fontsize=12)
        ax1.set_ylabel("Cloud Cover (%)", fontsize=12)
        ax1.set_title("Temporal Distribution of Products by Cluster", fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Format dates on x-axis
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

        # Plot 2: Monthly product count by cluster
        df["month"] = df["date"].dt.to_period("M")
        monthly_counts = df.groupby(["month", "label"]).size().unstack(fill_value=0)

        # Create stacked bar plot
        colors_dict = {
            label: self.class_dict.get(label, ("#808080", f"Cluster_{label}"))[0]
            for label in unique_labels
        }

        monthly_counts.plot(
            kind="bar",
            stacked=True,
            ax=ax2,
            color=[colors_dict[label] for label in monthly_counts.columns],
        )

        ax2.set_xlabel("Month", fontsize=12)
        ax2.set_ylabel("Number of Products", fontsize=12)
        ax2.set_title("Monthly Product Count by Cluster", fontsize=14)
        ax2.legend(title="Cluster")
        ax2.grid(True, alpha=0.3)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig, (ax1, ax2)

    def plot_cluster_statistics(self, figsize=(12, 8), save_path=None):
        """Plot statistics about each cluster"""
        if not self.products:
            raise ValueError("No products available")

        # Calculate statistics for each cluster
        cluster_stats = {}

        for label in np.unique(self.labels):
            mask = self.labels == label
            cluster_products = [
                self.products[i] for i in range(len(self.products)) if mask[i]
            ]
            cluster_thumbnails = np.array(self.thumbnails)[mask]

            stats = {
                "count": len(cluster_products),
                "mean_cloud_cover": np.mean([p.cloud_cover for p in cluster_products]),
                "std_cloud_cover": np.std([p.cloud_cover for p in cluster_products]),
                "mean_brightness": np.mean(cluster_thumbnails),
                "satellites": {},
            }

            # Count by satellite type
            for product in cluster_products:
                sat_type = product.satellite
                stats["satellites"][sat_type] = stats["satellites"].get(sat_type, 0) + 1

            cluster_name = self.class_dict.get(label, (None, f"Cluster_{label}"))[1]
            cluster_stats[cluster_name] = stats

        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # Plot 1: Product count by cluster
        names = list(cluster_stats.keys())
        counts = [stats["count"] for stats in cluster_stats.values()]
        colors = [
            self.class_dict.get(label, ("#808080", ""))[0]
            for label in np.unique(self.labels)
        ]

        axes[0, 0].bar(names, counts, color=colors)
        axes[0, 0].set_title("Product Count by Cluster")
        axes[0, 0].set_ylabel("Number of Products")
        plt.setp(axes[0, 0].xaxis.get_majorticklabels(), rotation=45)

        # Plot 2: Mean cloud cover by cluster
        cloud_means = [stats["mean_cloud_cover"] for stats in cluster_stats.values()]
        cloud_stds = [stats["std_cloud_cover"] for stats in cluster_stats.values()]

        axes[0, 1].bar(names, cloud_means, yerr=cloud_stds, color=colors, alpha=0.7)
        axes[0, 1].set_title("Mean Cloud Cover by Cluster")
        axes[0, 1].set_ylabel("Cloud Cover (%)")
        plt.setp(axes[0, 1].xaxis.get_majorticklabels(), rotation=45)

        # Plot 3: Mean brightness by cluster
        brightness_means = [
            stats["mean_brightness"] for stats in cluster_stats.values()
        ]

        axes[1, 0].bar(names, brightness_means, color=colors)
        axes[1, 0].set_title("Mean Thumbnail Brightness by Cluster")
        axes[1, 0].set_ylabel("Brightness (0-255)")
        plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45)

        # Plot 4: Satellite type distribution
        # Create a stacked bar chart for satellite types
        all_satellites = set()
        for stats in cluster_stats.values():
            all_satellites.update(stats["satellites"].keys())
        all_satellites = list(all_satellites)

        satellite_data = {}
        for sat in all_satellites:
            satellite_data[sat] = [
                stats["satellites"].get(sat, 0) for stats in cluster_stats.values()
            ]

        bottom = np.zeros(len(names))
        for i, sat in enumerate(all_satellites):
            axes[1, 1].bar(
                names, satellite_data[sat], bottom=bottom, label=sat, alpha=0.8
            )
            bottom += satellite_data[sat]

        axes[1, 1].set_title("Satellite Type Distribution by Cluster")
        axes[1, 1].set_ylabel("Number of Products")
        axes[1, 1].legend()
        plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig, axes, cluster_stats

    def create_cloudcover_meshgrid(self, ax, resolution=50, alpha=0.3):
        """Create a cloud cover background meshgrid"""
        if not self.products:
            return

        # Get data bounds
        x_min, x_max = (
            self.transformed_data[:, 0].min(),
            self.transformed_data[:, 0].max(),
        )
        y_min, y_max = (
            self.transformed_data[:, 1].min(),
            self.transformed_data[:, 1].max(),
        )

        # Create grid
        x_range = x_max - x_min
        y_range = y_max - y_min
        xx, yy = np.meshgrid(
            np.linspace(x_min - x_range * 0.1, x_max + x_range * 0.1, resolution),
            np.linspace(y_min - y_range * 0.1, y_max + y_range * 0.1, resolution),
        )

        # Interpolate cloud cover values to create smooth background
        from scipy.interpolate import griddata

        try:
            # Get cloud cover values and coordinates
            cloud_values = np.array([p.cloud_cover for p in self.products])
            coords = self.transformed_data

            # Interpolate cloud cover across the grid - always use first 2 components
            grid_values = griddata(
                coords[:, :2],  # Always use first 2 components for 2D interpolation
                cloud_values,
                (xx, yy),
                method="linear",  # Linear is more stable for scattered data
                fill_value=np.mean(cloud_values),
            )

            # Create contour plot
            contour = ax.contourf(
                xx, yy, grid_values, levels=20, alpha=alpha, cmap="Blues_r"
            )

            return contour

        except Exception as e:
            print(f"Could not create meshgrid: {e}")
            return None

    def plot_publication_quality(
        self,
        method_name="PCA",
        show_thumbnails=True,
        show_meshgrid=True,
        thumbnail_sample="cluster",
        figsize=(7, 4),
        dpi=300,
        save_path=None,
    ):  # Figure size for A4 page
        """
        Create publication-quality plot with all enhancements

        Args:
            method_name: Name of dimensionality reduction method
            show_thumbnails: Whether to overlay thumbnails
            show_meshgrid: Whether to show cloud cover background
            thumbnail_sample: 'cluster', 'random', or 'all'
            figsize: Figure size
            dpi: Resolution for saving
            save_path: Path to save the plot
        """
        if self.transformed_data is None:
            raise ValueError("No data available")

        # Create figure with high DPI
        fig, ax = plt.subplots(
            figsize=figsize, dpi=100
        )  # matplotlib will scale for save

        # Add cloud cover meshgrid background if requested
        if show_meshgrid:
            contour = self.create_cloudcover_meshgrid(ax, alpha=0.2)
            if contour:
                # Add colorbar for cloud cover
                cbar = plt.colorbar(contour, ax=ax, shrink=0.8, pad=0.02)
                cbar.set_label("Cloud Cover (%)", fontsize=12, fontweight="bold")
                cbar.ax.tick_params(labelsize=10)

        # Get unique labels and colors
        unique_labels = np.unique(self.labels)
        colors = [
            "#2ca02c",
            "#ff7f0e",
            "#d62728",
            "#1f77b4",
            "#9467bd",
            "#8c564b",
            "#e377c2",
        ]

        # Plot scatter points
        scatter_handles = []
        for i, label in enumerate(unique_labels):
            mask = self.labels == label
            label_name = self.class_dict.get(label, (None, f"Cluster {label}"))[1]

            scatter = ax.scatter(
                self.transformed_data[mask, 0],
                self.transformed_data[mask, 1],
                c=colors[i % len(colors)],
                label=f"{label_name} ({sum(mask)})",
                s=80 if not show_thumbnails else 40,
                alpha=0.8 if not show_thumbnails else 0.6,
                edgecolors="white",
                linewidth=1.5,
                zorder=3,
            )
            scatter_handles.append(scatter)

        # Add thumbnails if requested
        if show_thumbnails and self.thumbnails:
            # Determine which thumbnails to show
            if thumbnail_sample == "cluster":
                # Sample evenly from each cluster
                indices = []
                max_per_cluster = max(1, 12 // len(unique_labels))  # Up to 12 total
                for label in unique_labels:
                    mask = self.labels == label
                    label_indices = np.where(mask)[0]
                    n_samples = min(max_per_cluster, len(label_indices))
                    if n_samples > 0:
                        sampled = np.random.choice(
                            label_indices, n_samples, replace=False
                        )
                        indices.extend(sampled)
            elif thumbnail_sample == "all":
                indices = range(len(self.thumbnails))
            else:  # random
                n_images = min(len(self.thumbnails), 12)
                indices = np.random.choice(
                    len(self.thumbnails), n_images, replace=False
                )

            zoom = 0.12  # Larger thumbnails for better visibility
            for i in indices:
                try:
                    thumbnail = self.thumbnails[i]

                    # Resize thumbnail
                    resized_img = resize(
                        thumbnail,
                        output_shape=(
                            int(thumbnail.shape[0] * zoom),
                            int(thumbnail.shape[1] * zoom),
                        ),
                        anti_aliasing=True,
                        preserve_range=True,
                    ).astype(thumbnail.dtype)

                    # Get cluster color for border
                    label = self.labels[i]
                    label_idx = list(unique_labels).index(label)
                    border_color = colors[label_idx % len(colors)]

                    imagebox = OffsetImage(resized_img, zoom=zoom)
                    ab = AnnotationBbox(
                        imagebox,
                        (self.transformed_data[i, 0], self.transformed_data[i, 1]),
                        frameon=True,
                        bboxprops=dict(
                            edgecolor=border_color,
                            linewidth=2,
                            facecolor="white",
                            alpha=0.9,
                        ),
                        zorder=4,
                    )
                    ax.add_artist(ab)
                except Exception as e:
                    print(f"Error adding thumbnail {i}: {e}")

        # Set axis limits with proper padding
        x_min, x_max = (
            self.transformed_data[:, 0].min(),
            self.transformed_data[:, 0].max(),
        )
        y_min, y_max = (
            self.transformed_data[:, 1].min(),
            self.transformed_data[:, 1].max(),
        )

        x_range = x_max - x_min
        y_range = y_max - y_min
        padding_x = x_range * 0.15
        padding_y = y_range * 0.15

        ax.set_xlim(x_min - padding_x, x_max + padding_x)
        ax.set_ylim(y_min - padding_y, y_max + padding_y)

        # Publication-quality formatting
        ax.set_xlabel(f"{method_name} Component 1", fontsize=14, fontweight="bold")
        ax.set_ylabel(f"{method_name} Component 2", fontsize=14, fontweight="bold")

        # Determine satellite type for title
        satellite_types = set(p.satellite for p in self.products)
        if len(satellite_types) == 1:
            sat_name = list(satellite_types)[0].title()
        else:
            sat_name = "Multi-satellite"

        title = f"{sat_name} Thumbnail Clustering ({method_name})"
        if show_meshgrid:
            title += "\nwith Cloud Cover Background"

        ax.set_title(title, fontsize=16, fontweight="bold", pad=20)

        # Enhanced legend
        legend = ax.legend(
            handles=scatter_handles,
            loc="upper right",
            fontsize=11,
            frameon=True,
            fancybox=True,
            shadow=True,
            framealpha=0.9,
            edgecolor="black",
        )
        legend.get_frame().set_linewidth(1.5)

        # Grid and styling
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.8)
        ax.tick_params(axis="both", labelsize=11, width=1.2, length=4)

        # Ensure square aspect ratio for better comparison
        ax.set_aspect("equal", adjustable="box")

        plt.tight_layout()

        if save_path:
            plt.savefig(
                save_path,
                dpi=dpi,
                bbox_inches="tight",
                facecolor="white",
                edgecolor="none",
            )
            print(f"✅ Saved: {save_path}")

        return fig, ax

    def generate_all_plots(self, output_dir="publication_plots", dpi=300):
        """Generate all possible visualization plots"""
        import os

        os.makedirs(output_dir, exist_ok=True)

        plots_generated = []

        # Get method name
        method_name = (
            self.processor.reducer.get_name()
            if hasattr(self.processor, "reducer")
            else "Unknown"
        )
        satellite_name = (
            list(set(p.satellite for p in self.products))[0]
            if self.products
            else "unknown"
        )

        # 1. Basic scatter plot
        fig1, ax1 = self.plot_clusters_scatter(
            save_path=f"{output_dir}/{satellite_name}_{method_name}_scatter.png"
        )
        plots_generated.append(f"{satellite_name}_{method_name}_scatter.png")

        # 2. Thumbnails without meshgrid
        fig2, ax2 = self.plot_publication_quality(
            method_name=method_name.split("(")[0],  # Clean method name
            show_thumbnails=True,
            show_meshgrid=False,
            save_path=f"{output_dir}/{satellite_name}_{method_name}_thumbnails.png",
            dpi=dpi,
        )
        plots_generated.append(f"{satellite_name}_{method_name}_thumbnails.png")

        # 3. Thumbnails with cloud cover meshgrid
        fig3, ax3 = self.plot_publication_quality(
            method_name=method_name.split("(")[0],
            show_thumbnails=True,
            show_meshgrid=True,
            save_path=f"{output_dir}/{satellite_name}_{method_name}_with_meshgrid.png",
            dpi=dpi,
        )
        plots_generated.append(f"{satellite_name}_{method_name}_with_meshgrid.png")

        # 4. Temporal distribution
        if len(self.products) > 3:
            fig4, axes4 = self.plot_temporal_distribution(
                save_path=f"{output_dir}/{satellite_name}_{method_name}_temporal.png"
            )
            plots_generated.append(f"{satellite_name}_{method_name}_temporal.png")

        # 5. Cluster statistics
        fig5, axes5, stats = self.plot_cluster_statistics(
            save_path=f"{output_dir}/{satellite_name}_{method_name}_statistics.png"
        )
        plots_generated.append(f"{satellite_name}_{method_name}_statistics.png")

        print(f"\n✅ Generated {len(plots_generated)} publication-quality plots:")
        for plot in plots_generated:
            print(f"   • {plot}")

        return plots_generated

