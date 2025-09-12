import os
import random

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image
from scipy.interpolate import griddata
from skimage.transform import resize
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

import ShallowLearn.ExtractMetadata as extract_meta
import ShallowLearn.FileProcessing as fp
from ShallowLearn.API_Utils import filter_by_indices, filter_by_label

# Transform functions moved to core.array_utils
from ShallowLearn.core.array_utils import LCE_multi

# LoadData module removed - will need to update these imports later
# from ShallowLearn.LoadData import LoadSentinel2L1C as load_sen2
# from ShallowLearn.LoadData import PVI_Dataloader
from ShallowLearn.io import load_image


# Temporary PVI (Preview Image Files) loader - can be updated for API loading later
class PVI_Dataloader:
    """
    Preview Image Files (PVI) data loader for ZIP files containing Sentinel-2 preview images.
    This is a temporary implementation that can be extended for API-based loading.
    """

    def __init__(self, data_source: str):
        self.data_source = data_source
        self.is_zip = data_source.endswith(".zip")

        if self.is_zip:
            try:
                import zipfile

                with zipfile.ZipFile(data_source, "r") as zip_ref:
                    # Find PVI files in the ZIP
                    pvi_files = [
                        f
                        for f in zip_ref.namelist()
                        if "PVI" in f and f.endswith(".jp2")
                    ]
                    if not pvi_files:
                        raise ValueError(f"No PVI files found in {data_source}")
                    self.files = pvi_files[0]  # Take first PVI file
            except Exception as e:
                print(
                    f"File: {data_source} failed. Please double check integrity of file"
                )
                raise e

        self.zip_path = f"zip+file://{data_source}/{self.files}"

    def load(self) -> np.ndarray:
        """Load Preview Image Files data from ZIP file."""
        import rasterio

        with rasterio.open(self.zip_path) as dataset:
            pvi_image = dataset.read()
            # Transpose from (bands, height, width) to (height, width, bands)
            pvi_image = np.transpose(pvi_image, (1, 2, 0))
        return pvi_image


from ShallowLearn.Util import clip_image


def plot_images_on_pca(
    transformed_data,
    original_images,
    zoom=0.1,
    figsize=(10, 10),
    title="Visualization of Transformed Imagery",
):
    """
    Plots images on their respective PCA-transformed coordinates.

    Parameters:
    - transformed_data: numpy array of shape (n_samples, 2) containing PCA-transformed coordinates.
    - original_images: numpy array of shape (n_samples, height, width, 3) containing the original images.
    - zoom: float, the zoom level of the images on the plot.
    - figsize: tuple, the size of the figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    for i in range(len(transformed_data)):
        original_dtype = original_images[i].dtype

        shape = original_images[i].shape
        # Adjust the call to `resize` to ensure it matches the original data type and range
        # Note: `anti_aliasing` is generally a good idea when downsampling images
        img = resize(
            original_images[i],
            output_shape=(int(shape[0] * zoom), int(shape[1] * zoom)),
            anti_aliasing=True,
            preserve_range=True,
        ).astype(original_dtype)
        # print(img.shape)
        # plt.imshow(img)
        # plt.show()
        # break
        imagebox = OffsetImage(np.array(img), zoom=zoom)
        ab = AnnotationBbox(
            imagebox, (transformed_data[i, 0], transformed_data[i, 1]), frameon=False
        )
        ax.add_artist(ab)

    ax.scatter(
        transformed_data[:, 0], transformed_data[:, 1], alpha=0.2
    )  # Plot transparent points to keep the scale
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    # plt.savefig("../Graphs/PVI_PCA2.svg")
    # plt.show()
    return ax


class QuickLookModel:
    """Abstract base class implementation do not use"""

    def __init__(self, files, model=None):
        self.PVI = None
        self.imagery = []
        self.pca_model = PCA()

    def __len__(self):
        return len(self.imagery)

    def create_custom_pastel_cmap(self, labels):
        """
        Create a custom colormap using a pastel theme for the given labels.

        Parameters:
        - labels: array-like of unique labels

        Returns:
        - custom_cmap: ListedColormap with tab20c colors
        """
        tab20_colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))
        custom_cmap = ListedColormap(tab20_colors)
        return custom_cmap

    def load_data(self):
        pass

    def train(self):
        transformed_imagery = (
            np.array(self.imagery).reshape(len(self.imagery), -1) / 255
        )
        transformed_data = self.pca_model.fit_transform(transformed_imagery)
        return transformed_data

    def predict(self, transformed_data, model=None, eps=50, min_samples=5):
        if model is None:
            dbscan_model = DBSCAN(eps=eps, min_samples=min_samples)
            dbscan_model.fit(transformed_data)
            return dbscan_model.labels_

    def generate_dataframe(self):
        raise Exception("Not implemented in baseclass")

    def plot_cloud_coverage(
        self,
        df,
        class_dict,
        zoom=0.05,
        save=None,
        show_images=True,
        cloud_threshold=50,
        percentage_show=75,
        ax=None,
        title=None,
    ):
        """
        Parameters:
        -----------
        class_dict: dict
            Dictionary mapping class numbers to (color, label) tuples
            e.g., {-1: ('#1f77b4', 'Clear Sky'),
                    0: ('#ff7f0e', 'Partially Cloudy'),
                    1: ('#2ca02c', 'Cloudy')}
        """
        # Filter data based on cloud coverage
        valid_indices = df["CLOUD_COVERAGE_ASSESSMENT"] <= cloud_threshold
        filtered_df = df[valid_indices].copy()
        filtered_imagery = [
            self.imagery[i] for i in range(len(self.imagery)) if valid_indices.iloc[i]
        ]

        # Create color map from dictionary
        label_list = sorted(filtered_df.Label.unique())  # Sort labels for consistency
        colors = [class_dict[label][0] for label in label_list]
        custom_cmap = ListedColormap(colors)

        # Convert datetime column
        filtered_df["DATATAKE_1_DATATAKE_SENSING_START"] = pd.to_datetime(
            filtered_df["DATATAKE_1_DATATAKE_SENSING_START"], errors="coerce"
        )

        if ax is None:
            show = True
            fig, ax = plt.subplots(figsize=(15, 15))
        else:
            show = False

        # Formatting date on x-axis
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_minor_formatter(mdates.DateFormatter("%b"))

        # Adding labels and title
        ax.set_xlabel("Sensing Start Date", fontsize=30)
        ax.set_ylabel("Cloud Coverage Assessment", fontsize=30)
        if title is None:
            ax.set_title(
                f"Cloud Coverage Assessment over Time (Cloud Coverage ≤ {cloud_threshold}%)",
                fontsize=30,
            )
        else:
            ax.set_title(f"{title}", fontsize=30)

        # Adding a grid
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)

        if show_images:
            # Adding images on the plot with a border corresponding to the label color
            for x, y, img, label in zip(
                filtered_df["DATATAKE_1_DATATAKE_SENSING_START"],
                filtered_df["CLOUD_COVERAGE_ASSESSMENT"],
                filtered_imagery,
                filtered_df["Label"],
            ):
                if not self.PVI:
                    img = img[:, :, [2, 1, 0]]
                num = random.randint(0, 100)
                if num < percentage_show:
                    continue
                imagebox = OffsetImage(img, zoom=zoom, alpha=0.783)
                label_idx = label_list.index(label)  # Get index of label in sorted list
                ab = AnnotationBbox(
                    imagebox,
                    (mdates.date2num(x), y),
                    frameon=True,
                    bboxprops=dict(edgecolor=custom_cmap(label_idx), linewidth=2),
                )
                ax.add_artist(ab)

        # Scatter plot with discrete colors
        scatter = ax.scatter(
            filtered_df["DATATAKE_1_DATATAKE_SENSING_START"],
            filtered_df["CLOUD_COVERAGE_ASSESSMENT"],
            c=[label_list.index(label) for label in filtered_df["Label"]],
            cmap=custom_cmap,
            edgecolor="black",
            linewidth=1,
            s=50,
        )

        # Adding a color bar with labels
        cbar = plt.colorbar(
            scatter, ax=ax, ticks=range(len(label_list)), fraction=0.046, pad=0.04
        )
        cbar.set_label("Class Label", fontsize=30)
        cbar.set_ticklabels([class_dict[label][1] for label in label_list])
        ax.tick_params(axis="both", labelsize=25)  # Changes both x and y axis numbers

        # plt.tight_layout()
        if save is not None:
            plt.savefig(save)
        if show:
            plt.show()

    def plot_principal_components(
        self,
        df,
        class_dict,
        plot_meshgrid=True,
        zoom=0.05,
        save=None,
        show_images=True,
        cloud_threshold=50,
        percentage_show=75,
        ax=None,
        title=None,
        axis_extent=10_000,
    ):
        """
        Parameters:
        -----------
        class_dict: dict
            Dictionary mapping class numbers to (color, label) tuples
            e.g., {-1: ('#1f77b4', 'Clear Sky'),
                    0: ('#ff7f0e', 'Partially Cloudy'),
                    1: ('#2ca02c', 'Cloudy')}
        """
        # First perform PCA on all imagery for the mesh
        full_pca_result = self.pca_model.transform(
            [img.flatten() for img in self.imagery]
        )

        # Then filter data based on cloud coverage
        valid_indices = df["CLOUD_COVERAGE_ASSESSMENT"] <= cloud_threshold
        filtered_df = df[valid_indices].copy()
        filtered_imagery = [
            self.imagery[i] for i in range(len(self.imagery)) if valid_indices.iloc[i]
        ]
        filtered_pca = full_pca_result[valid_indices]

        # Create color map from dictionary
        label_list = sorted(filtered_df.Label.unique())  # Sort labels for consistency
        colors = [class_dict[label][0] for label in label_list]
        custom_cmap = ListedColormap(colors)

        background_cmap = plt.cm.viridis

        # Plotting
        if ax is None:
            show = True
            fig, ax = plt.subplots(figsize=(15, 15))
        else:
            show = False

        if plot_meshgrid:
            # Use full dataset for mesh boundaries
            x_min, x_max = (
                full_pca_result[:, 0].min() - axis_extent,
                full_pca_result[:, 0].max() + axis_extent,
            )
            y_min, y_max = (
                full_pca_result[:, 1].min() - axis_extent,
                full_pca_result[:, 1].max() + axis_extent,
            )
            xx, yy = np.meshgrid(
                np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100)
            )

            grid_points = np.c_[xx.ravel(), yy.ravel()]
            dummy_points = np.zeros(
                (grid_points.shape[0], self.pca_model.n_components_)
            )
            dummy_points[:, :2] = grid_points

            # inverse_transformed_points = self.pca_model.inverse_transform(dummy_points)
            cloud_coverage_grid = griddata(
                full_pca_result[:, :2],
                df["CLOUD_COVERAGE_ASSESSMENT"],
                (xx, yy),
                method="linear",
            )

            c = ax.imshow(
                cloud_coverage_grid,
                extent=(x_min, x_max, y_min, y_max),
                origin="lower",
                cmap=background_cmap,
                alpha=0.5,
            )

            cbar_bg = plt.colorbar(c, ax=ax, fraction=0.046, pad=0.04)
            cbar_bg.set_label("Cloud Coverage Assessment", fontsize=30)
            cbar_bg.ax.tick_params(labelsize=25)  # Adjust 20 to your desired font size

        ax.set_xlabel("Principal Component 1", fontsize=30)
        ax.set_ylabel("Principal Component 2", fontsize=30)
        if title is None:
            ax.set_title(
                f"Principal Component Analysis of Imagery (Cloud Coverage ≤ {cloud_threshold}%)",
                fontsize=30,
            )
        else:
            ax.set_title(f"{title}", fontsize=30)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)

        if show_images:
            for pc1, pc2, img, label in zip(
                filtered_pca[:, 0],
                filtered_pca[:, 1],
                filtered_imagery,
                filtered_df["Label"],
            ):
                if not self.PVI:
                    img = img[:, :, [2, 1, 0]]
                num = random.randint(0, 100)
                if num < percentage_show:
                    continue
                imagebox = OffsetImage(img, zoom=zoom, alpha=0.9)
                label_idx = label_list.index(label)  # Get index of label in sorted list
                ab = AnnotationBbox(
                    imagebox,
                    (pc1, pc2),
                    frameon=True,
                    bboxprops=dict(edgecolor=custom_cmap(label_idx), linewidth=2),
                    box_alignment=(0.5, 0.5),
                )
                ax.add_artist(ab)

        scatter = ax.scatter(
            filtered_pca[:, 0],
            filtered_pca[:, 1],
            c=[label_list.index(label) for label in filtered_df["Label"]],
            cmap=custom_cmap,
            edgecolor="black",
            linewidth=1,
            s=50,
        )

        # Ensure the colorbar ticks match the actual labels
        cbar = plt.colorbar(
            scatter, ax=ax, ticks=range(len(label_list)), fraction=0.01, pad=0.01
        )
        cbar.set_ticklabels([class_dict[label][1] for label in label_list])
        cbar.ax.tick_params(labelsize=25)  # Adjust 20 to your desired font size

        ax.set_xlim(
            [
                filtered_pca[:, 0].min() - axis_extent,
                filtered_pca[:, 0].max() + axis_extent,
            ]
        )
        ax.set_ylim(
            [
                filtered_pca[:, 1].min() - axis_extent,
                filtered_pca[:, 1].max() + axis_extent,
            ]
        )
        ax.tick_params(axis="both", labelsize=25)  # Changes both x and y axis numbers
        # plt.tight_layout()
        if save is not None:
            plt.savefig(save)
        if show:
            plt.show()


class QuickLookPVI(QuickLookModel):
    def __init__(self, files, model=None):
        super().__init__(files, model)
        self.PVI = True
        # Initialize load_zips default
        self.load_zips = False

        if len(files) <= 1 and os.path.isdir(files):
            files = fp.extract_pvi_images(files)
            self.load_zips = False
        elif len(files) > 1 and files[0].endswith(".zip"):
            self.load_zips = True
        elif isinstance(files, list):
            print("Starting PCA Model")
            self.load_zips = False  # Default for list of files
        else:
            raise ValueError("Need to add a path or a list of files to use method")
        components = 4
        if model is None:
            self.pca_model = PCA(n_components=components)
        self.files = files
        self.imagery = self.load_data()
        self.transformed_data = self.train()
        self.labels = self.predict(self.transformed_data)

    def load_data(self):
        imagery = []
        files = []
        if self.load_zips is False:
            for file in self.files:
                try:
                    with Image.open(file) as im:
                        imagery.append(np.array(im))
                        files.append(file)  # Add successfully loaded files
                except Exception as e:
                    print(f"File Failed to load {file}: {e}")
        else:
            for file in self.files:
                try:
                    img = PVI_Dataloader(file).load()
                    imagery.append(img)
                    files.append(file)
                except Exception as e:
                    print(f"File Failed to load {file}: {e}")
        self.files = files  # Update files to only include successfully loaded ones
        return imagery

    def generate_dataframe(self, directory, zips=False):
        if zips:
            return extract_meta.combine_metadata_w_pvi_analysis(
                self.files, self, gen_from_zips=zips
            )
        return extract_meta.combine_metadata_w_pvi_analysis(
            directory, self, gen_from_zips=zips
        )


class QuickLookAPIPVI(QuickLookModel):
    def __init__(self, features, thumbnail_dir, download_thumbnails=False, model=None):
        self.PVI = True
        self.features = features
        self.df = extract_meta.generate_api_df(
            features, thumbnail_dir, download_thumbnails=download_thumbnails
        )
        self.column_updated_df = extract_meta.map_and_filter_columns(self.df)
        self.thumbnails_dir = thumbnail_dir
        self.files = []
        self.imagery = []
        self.load_data()
        if model is None:
            self.pca_model = PCA(n_components=0.95)
        self.transformed_data = self.train()
        self.labels = self.predict(self.transformed_data, min_samples=5, eps=50)
        self.column_updated_df["Label"] = self.labels
        self.generate_class_dict()

    def predict(self, transformed_data, model=None, eps=50, min_samples=5):
        if model is None:
            dbscan_model = DBSCAN(eps=eps, min_samples=min_samples)
            dbscan_model.fit(transformed_data)
            self.column_updated_df["Label"] = dbscan_model.labels_
            return dbscan_model.labels_

    def load_data(self):
        self.imagery = []
        self.files = []
        valid_indices = []

        for idx, component in enumerate(self.column_updated_df.COMMON_COMPONENT):
            file_path = os.path.join(self.thumbnails_dir, component + ".jpg")
            self.files.append(file_path)

            try:
                with Image.open(file_path) as im:
                    img_array = np.array(im)
                    if img_array.shape == (343, 343, 3):
                        self.imagery.append(img_array)
                        valid_indices.append(idx)
                    else:
                        print(f"Skipping {file_path}: Invalid shape {img_array.shape}")
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")

        # Keep only valid rows in the DataFrame
        self.column_updated_df = self.column_updated_df.iloc[valid_indices].reset_index(
            drop=True
        )
        self.files = [self.files[i] for i in valid_indices]
        self.features = filter_by_indices(self.features, valid_indices)

    def get_filtered_features(self, label):
        return filter_by_label(self.features, self.labels, label)

    def generate_class_dict(self):
        mean_0 = np.array(self.imagery)[self.column_updated_df.Label == 0].mean()
        mean_1 = np.array(self.imagery)[self.column_updated_df.Label == 1].mean()

        class_dict = {-1: ("#1f77b4", "Partially Cloudy"), 2: ("#d62728", "No Data")}

        if mean_0 > mean_1:
            class_dict.update(
                {0: ("#2ca02c", "Opaque Clouds"), 1: ("#ff7f0e", "Clear Sky")}
            )
        else:
            class_dict.update(
                {0: ("#ff7f0e", "Clear Sky"), 1: ("#2ca02c", "Opaque Clouds")}
            )
        self.class_dict = class_dict

    def get_cloud_free_dataset(self):
        cloud_key = [k for k, v in self.class_dict.items() if v[1] == "Clear Sky"][0]
        return self.get_filtered_features(cloud_key)


class QuickLookArea(QuickLookModel):
    def __init__(
        self,
        df,
        shapefile,
        band_mapping=["B02", "B03", "B04", "B08"],
        resolution="10m",
        stretch_type=LCE_multi,
    ):
        self.df = df
        self.stretch_type = stretch_type
        # self.df = df[(df.Label == 0) | (df.Label == -1)]
        self.files = self.df.FILE_PATH.to_list()
        self.PVI = False
        # print(self.files)
        # self.files = fp.extract_MTD_files(directory)
        self.shapefile = shapefile
        self.imagery = self.load_data(band_mapping, resolution)
        print("Data loading finished")
        components = 0.95
        self.pca_model = PCA(n_components=components)
        self.df = self.filter_dataframe_by_file_path(df, self.files)
        self.transformed_data = self.train()
        self.labels = self.predict()
        self.df["Label"] = self.labels

    def train(self):
        transformed_imagery = np.array(self.imagery).reshape(len(self.imagery), -1)
        transformed_data = self.pca_model.fit_transform(transformed_imagery)
        return transformed_data

    def load_data(self, band_mapping=["B02", "B03", "B04", "B08"], resolution="10m"):
        imagery = []
        self.updated_files = []
        for file in tqdm(self.files, desc="Processing files"):
            # print(file)
            image = load_image(file)
            clipped = image.clip_raster_with_shape(
                self.shapefile, resolution, selected_bands=band_mapping, use_mask=False
            )
            clipped -= 1000
            try:
                clipped = np.swapaxes(clipped, 0, 2)
                if self.stretch_type is not None:
                    clipped = self.stretch_type(clipped) / 255

                clipped = clip_image(clipped, clip_percent=2)
                # clipped = trf.LCE_multi(clipped)
                imagery.append(clipped)
                self.updated_files.append(file)
            except:
                print(f"{file} failed to transform")

        self.files = self.updated_files
        return imagery

    def generate_dataframe(self):
        return self.df

    def filter_dataframe_by_file_path(self, df, valid_file_paths):
        """
        Drops rows from a DataFrame based on values in the 'FILE_PATH' column
        that are not present in the provided list of valid file paths.

        Parameters:
        - df: The DataFrame to filter.
        - valid_file_paths: A list of valid file paths. Rows with 'FILE_PATH' not in this list will be dropped.

        Returns:
        - A filtered DataFrame with only rows that have 'FILE_PATH' values in the valid_file_paths list.
        """
        filtered_df = df[df["FILE_PATH"].isin(valid_file_paths)]
        return filtered_df

    def predict(self, model=None, xi=0.05, min_samples=30, min_cluster_size=0.01):
        if model is None:
            clust = GaussianMixture(n_components=4, random_state=42)
            clust.fit_predict(self.transformed_data)

            labels = clust.fit_predict(self.transformed_data)
            self.df["Label"] = labels
            return self.df["Label"]
