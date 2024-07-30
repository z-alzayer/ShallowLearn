import numpy as np
import os
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.dates as mdates
from PIL import Image
from sklearn.decomposition import PCA, KernelPCA
from sklearn.cluster import DBSCAN
from tqdm import tqdm


import ShallowLearn.FileProcessing as fp
from ShallowLearn.LoadData import LoadSentinel2L1C as load_sen2
import ShallowLearn.ExtractMetadata as extract_meta
from ShallowLearn.band_mapping import band_mapping
from ShallowLearn.Util import clip_image
import ShallowLearn.Transform as trf

from skimage.transform import resize
from sklearn.mixture import GaussianMixture
from scipy.interpolate import griddata

def plot_images_on_pca(transformed_data, original_images, zoom=0.1, figsize=(10, 10), title = "Visualization of Transformed Imagery"):
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
        img = resize(original_images[i], 
                     output_shape=(int(shape[0] * zoom), int(shape[1] * zoom)), 
                     anti_aliasing=True, preserve_range=True).astype(original_dtype)
        # print(img.shape)
        # plt.imshow(img)
        # plt.show()
        # break
        imagebox = OffsetImage(np.array(img), zoom=zoom)
        ab = AnnotationBbox(imagebox, (transformed_data[i, 0], transformed_data[i, 1]), frameon=False)
        ax.add_artist(ab)
    
    ax.scatter(transformed_data[:, 0], transformed_data[:, 1], alpha=0.2)  # Plot transparent points to keep the scale
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(title)
    # plt.savefig("../Graphs/PVI_PCA2.svg")
    # plt.show()
    return ax


class QuickLookModel():
    """Abstract base class implementation do not use"""
    def __init__(self, files, model = None):
        self.PVI = None



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
        transformed_imagery = np.array(self.imagery).reshape(len(self.imagery), -1)/255
        transformed_data = self.pca_model.fit_transform(transformed_imagery)
        return transformed_data

    def predict(self, transformed_data, model = None):
        if model is None:
            dbscan_model = DBSCAN(eps=30, min_samples=9)
            dbscan_model.fit(transformed_data)
            return dbscan_model.labels_ 
    
    def generate_dataframe():
        raise Exception("Not implemented in baseclass")


    def plot_cloud_coverage(self, df, zoom = 0.05):
        if self.PVI:
            custom_cmap = ListedColormap(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])  # Four distinct colors
        else:
            label_list = (df.Label.unique())
            custom_cmap = self.create_custom_pastel_cmap(label_list)
        df['DATATAKE_1_DATATAKE_SENSING_START'] = pd.to_datetime(df['DATATAKE_1_DATATAKE_SENSING_START'], errors='coerce')

        # Plotting
        fig, ax = plt.subplots(figsize=(12, 6))

        # Formatting date on x-axis
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_minor_formatter(mdates.DateFormatter('%b'))

        # Adding labels and title
        ax.set_xlabel('Sensing Start Date', fontsize=14)
        ax.set_ylabel('Cloud Coverage Assessment', fontsize=14)
        ax.set_title('Cloud Coverage Assessment over Time', fontsize=16)

        # Adding a grid
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Adding images on the plot with a border corresponding to the label color
        for x, y, img, label in zip(df['DATATAKE_1_DATATAKE_SENSING_START'], df['CLOUD_COVERAGE_ASSESSMENT'], self.imagery, df['Label']):
            if self.PVI == False:
                img = img[:,:,[2,1,0]]
            imagebox = OffsetImage(img, zoom=zoom, alpha=0.783)  # Adjust the zoom parameter and alpha for transparency
            ab = AnnotationBbox(imagebox, (mdates.date2num(x), y),
                                frameon=True,
                                bboxprops=dict(edgecolor=custom_cmap(label+1), linewidth=2))  # Offset label by 1 to handle -1
            ax.add_artist(ab)

        # Scatter plot with discrete colors on top of the images
        scatter = ax.scatter(df['DATATAKE_1_DATATAKE_SENSING_START'], df['CLOUD_COVERAGE_ASSESSMENT'], 
                             c=df['Label'], cmap=custom_cmap, edgecolor='black', linewidth=1, s=50)

        # Adding a color bar with discrete labels
        cbar = plt.colorbar(scatter, ax=ax, ticks=np.arange(-1, 3))
        cbar.set_label('Label', fontsize=12)
        if self.PVI:
            cbar.set_ticks([-1, 0, 1, 2])  # Ensure all four labels are present
            cbar.set_ticklabels(['Partially Cloudy', 'Clear Sky', 'Opaque Clouds', 'No Data'])  # Custom labels for classes

        # Improving the overall appearance
        plt.tight_layout()

        # Show plot
        plt.show()
        
    def plot_principal_components(self, df, plot_meshgrid=True, zoom = 0.05):
        if self.PVI:
            custom_cmap = ListedColormap(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])  # Four distinct colors
            num = 1
        else:
            label_list = (df.Label.unique())
            custom_cmap = self.create_custom_pastel_cmap(label_list)
            num = 0
        background_cmap = plt.cm.viridis  # Warm and cool colormap

        # Perform PCA on the imagery data
        pca_result = self.pca_model.transform([img.flatten() for img in self.imagery])

        # Plotting
        fig, ax = plt.subplots(figsize=(15, 15))

        if plot_meshgrid:
            # Create a meshgrid for the background
            x_min, x_max = pca_result[:, 0].min() - 1, pca_result[:, 0].max() + 1
            y_min, y_max = pca_result[:, 1].min() - 1, pca_result[:, 1].max() + 1
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

            # Flatten the meshgrid arrays and create a PCA input format
            grid_points = np.c_[xx.ravel(), yy.ravel()]

            # Create a dummy array with the correct number of components
            dummy_points = np.zeros((grid_points.shape[0], self.pca_model.n_components_))
            dummy_points[:, :2] = grid_points

            # Inverse transform to the original space
            inverse_transformed_points = self.pca_model.inverse_transform(dummy_points)

            # Predict or interpolate the cloud coverage for the grid points
            cloud_coverage_grid = griddata(pca_result[:, :2], df['CLOUD_COVERAGE_ASSESSMENT'], (xx, yy), method='linear')

            # Plot the background meshgrid
            c = ax.imshow(cloud_coverage_grid, extent=(x_min, x_max, y_min, y_max), origin='lower', cmap=background_cmap, alpha=0.5)

            # Adding a color bar for the background
            cbar_bg = plt.colorbar(c, ax=ax, fraction=0.01, pad = 0.1)
            cbar_bg.set_label('Cloud Coverage Assessment', fontsize=12)

        # Adding labels and title
        ax.set_xlabel('Principal Component 1', fontsize=14)
        ax.set_ylabel('Principal Component 2', fontsize=14)
        ax.set_title('Principal Component Analysis of Imagery with Cloud Coverage', fontsize=16)

        # Adding a grid
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Adding images on the plot with a border corresponding to the label color
        for pc1, pc2, img, label in zip(pca_result[:, 0], pca_result[:, 1], self.imagery, df['Label']):
            if self.PVI == False:
                img = img[:,:,[2,1,0]]
            imagebox = OffsetImage(img, zoom=zoom, alpha=0.9)  # Adjust the zoom parameter and alpha for transparency
            ab = AnnotationBbox(imagebox, (pc1, pc2),
                                frameon=True,
                                bboxprops=dict(edgecolor=custom_cmap(label + num), linewidth=2), 
                                box_alignment=(0.5, 0.5))  # Align images to the center
            ax.add_artist(ab)

        # Scatter plot with discrete colors on top of the images
        scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], 
                            c=df['Label'], cmap=custom_cmap, edgecolor='black', linewidth=1, s=50)

        # Adding a color bar with discrete labels
        cbar = plt.colorbar(scatter, ax=ax, ticks=np.arange(-1, 3), fraction=0.01, pad=0.01)
        # cbar.set_label('Label', fontsize=12)
        if self.PVI:
            cbar.set_ticks([-1, 0, 1, 2])  # Ensure all four labels are present
            cbar.set_ticklabels(['Partially Cloudy', 'Clear Sky', 'Opaque Clouds', 'No Data'])  # Custom labels for classes
        else:
            cbar.set_ticks(df.Label.unique())
        # Ensure images at the border are either partially hidden or within the figure
        ax.set_xlim([pca_result[:, 0].min() - 1, pca_result[:, 0].max() + 1])
        ax.set_ylim([pca_result[:, 1].min() - 1, pca_result[:, 1].max() + 1])

        # Improving the overall appearance
        plt.tight_layout()

        # Show plot
        plt.show()
class QuickLookPVI(QuickLookModel):

    def __init__(self, files, model = None):
        super().__init__(files, model)
        self.PVI = True
        if os.path.isdir(files):
            files = fp.extract_pvi_images(files)
            # print(files)
        elif isinstance(files, list):
            print("Starting PCA Model")
        else:
            raise ValueError("Need to add a path or a list of files to use method")
        components = 4
        if model is None:
            self.pca_model = PCA(n_components = components)  
        self.files = files
        self.imagery = self.load_data()
        self.transformed_data = self.train()
        self.labels = self.predict(self.transformed_data)

    def load_data(self):
        imagery = []
        for file in self.files:
            with Image.open(file) as im:
                imagery.append(np.array(im))
        return imagery
    
    def generate_dataframe(self, directory):
        return extract_meta.combine_metadata_w_pvi_analysis(directory, self)


class QuickLookArea(QuickLookModel):
    
    def __init__(self, df, shapefile, 
                 band_mapping =  ['B02', 'B03', 'B04', 'B08'], resolution = "10m", stretch_type = trf.LCE_multi):
        self.df = df[df.Label == 0]
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
        self.pca_model = PCA(n_components = components)  
        self.transformed_data = self.train()
        self.labels = self.predict()
        self.df['Label'] = self.labels

    def load_data(self, band_mapping = ['B02', 'B03', 'B04', 'B08'], resolution = "10m"):
        imagery = []
        self.updated_files = []
        for file in tqdm(self.files, desc="Processing files"):
            print(file)
            image = load_sen2(file)
            clipped = image.clip_raster_with_shape(self.shapefile, resolution,
                                                   selected_bands=band_mapping, use_mask=False)
            # bit hacky - fix in dataloader
            if "N0400" in file:
                clipped -= 1000
            try:
                clipped = np.swapaxes(clipped, 0, 2)
                if self.stretch_type is not None:
                    clipped = self.stretch_type(clipped)
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

    def predict(self, model = None, n_components = 4):
        if model is None:
            model = GaussianMixture(n_components=n_components, random_state=42)
            
            self.df['Label'] = model.fit_predict(self.transformed_data)
            return self.df['Label']
         
if __name__ == "__main__":
    path = "/mnt/sda_mount/All_L1C_55LCD/"
    # PVI_Files = fp.extract_pvi_images(path)
    # print(PVI_Files)
    limits  = ((-14.4637,145.1483),(-15.4559,146.1532))
    from rasterio.warp import transform_bounds
    wcmc = gpd.read_file("Data/14_001_WCMC008_CoralReefs2018_v4_1/01_Data/WCMC008_CoralReef2018_Py_v4_1.shp")
    file = '/mnt/sda_mount/All_L1C_55LCD/S2B_MSIL1C_20220406T003659_N0400_R059_T55LCD_20220406T015555.SAFE/MTD_MSIL1C.xml'

    data_loader = load_sen2(file)
    data_loader.load()
    limits = transform_bounds(32755 ,f'EPSG:4326', *data_loader.bounds)
    subset_limits = ((limits[1], limits[0]), (limits[3], limits[2])) 
    top_left = subset_limits[0]
    bottom_right = subset_limits[1]

    subset = wcmc.cx[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

    subset = subset.to_crs(32755) 
    reef = subset.sample(1, random_state = 42)
    # image = data_loader.clip_raster_with_shape(reef, selected_bands= ['B04','B03','B02'], use_mask=False)


    print("Retrieved shapefile")




    qck_look = QuickLookArea(path, reef)
    print(qck_look.labels)








#     import timeit
#     setup_code = """
# import ShallowLearn.FileProcessing as fp

# path = "/mnt/sda_mount/All_L1C_55LCD/"
# PVI_Files = fp.extract_pvi_images(path)
#     """

#     test_code = """
# from ShallowLearn.QuickLook import QuickLookPCA
# qck_look = QuickLookPCA(PVI_Files)
# print(qck_look.labels)
#     """

#     # Number of times to run the code
#     num_runs = 10

#     # Time the execution
#     times = timeit.repeat(stmt=test_code, setup=setup_code, repeat=num_runs, number=1)

#     # Print the times
#     for i, t in enumerate(times):
#         print(f"Run {i + 1}: {t:.6f} seconds")

#     print(f"Average time: {sum(times) / len(times):.6f} seconds")
