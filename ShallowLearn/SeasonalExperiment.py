import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.cm import ScalarMappable
from matplotlib.patches import FancyArrowPatch

import ShallowLearn.LoadData as load_data
import ShallowLearn.LabelLoader as label_loader
import ShallowLearn.ImageHelper as ih
import ShallowLearn.Transform as trf
from ShallowLearn.PreprocDecorators import remove_zeros_from_image
import numpy as np
import matplotlib.pyplot as plt
import ShallowLearn.ComputerVisionFeatures as cvf
import pandas as pd
from ShallowLearn import band_mapping


path = "/mnt/sda_mount/Clipped/L1C/"
season = 'Winter'

def load_seasonal_data(path, substr = "/44_", season = 'Winter'):
    seasonal_loader = load_data.LoadSeasonalData(path, substr)
    dates = seasonal_loader.generate_dates()
    seasonal_images = seasonal_loader.gen_seasonal_images()
    images = seasonal_loader.load_seasonal_images(f"{season}")
    median_images = np.ma.median(images, axis = 0)
    return median_images, images, dates

median_images, images, dates = load_seasonal_data(path, substr = "/44_", season = 'Winter')
# fig,ax = plt.subplots(1,1, figsize=(10,10))

median_images_filled = median_images.filled(np.nan)
# plt.imshow(ih.plot_rgb(median_images_filled))
# plt.show()


def generate_features(image):
    texture = cvf.texture_features(image, P=8, R=30)
    gabor = cvf.gabor_features(image, frequency = 2.5)
    hog = cvf.histogram_of_oriented_gradients(image)
    sobel = cvf.sobel_edge_detection(image)
    canny = cvf.edge_density(image)
    arr = np.array((texture, gabor, hog, sobel, canny))
    arr = np.swapaxes(arr, 0, 1)
    arr = np.swapaxes(arr, 1, 2)
    return arr


def gen_color_spaces(image):
    lab = ih.plot_lab((image))
    hsv = ih.plot_hsv((image))
    return np.concatenate((lab, hsv), axis = 2)

feature_names = ['texture', 'gabor', 'hog', 'sobel', 'canny']
features = generate_features(ih.plot_rgb(median_images))
# plt.show()
color_names = ['l','a','b', 'h', 's', 'v']
color_spaces = gen_color_spaces(median_images)

import ShallowLearn.IndiceFeatures as indice_features
median_image = trf.LCE_multi(remove_zeros_from_image(median_images))

indice_object = indice_features.GenerateIndicesPerImage(median_images)


def create_stack(img):
    features = generate_features(ih.plot_rgb(img))
    indice_object = indice_features.GenerateIndicesPerImage(img)
    color_stack = gen_color_spaces(img)
    print("shapes====================")
    print(img.shape)
    print(features.shape)
    print(indice_object.indices.shape)
    print(color_stack.shape)
    print("====================")
    feature_stack = np.concatenate((img, features, color_stack, indice_object.indices), axis = 2)
    return feature_stack

feature_stack = create_stack(median_images)
print(feature_stack.shape)

# Assuming 'feature_images' is your array stack with shape (27, height, width)
# and 'features' is your list of feature names
n_features = feature_stack.shape[2]
grid_rows = 6
grid_cols = 6

feature_names = list(band_mapping.band_mapping.keys()) + feature_names + color_names + indice_object.indice_order


feature_properties = {
    'B01': {'cmap': 'gray', 'scale': (0, 1), 'title': 'B01'},
    'B02': {'cmap': 'gray', 'scale': (0, 1), 'title': 'B02'},
    'B03': {'cmap': 'gray', 'scale': (0, 1), 'title': 'B03'},
    'B04': {'cmap': 'gray', 'scale': (0, 1), 'title': 'B04'},
    'B05': {'cmap': 'gray', 'scale': (0, 1), 'title': 'B05'},
    'B06': {'cmap': 'gray', 'scale': (0, 1), 'title': 'B06'},
    'B07': {'cmap': 'gray', 'scale': (0, 1), 'title': 'B07'},
    'B08': {'cmap': 'gray', 'scale': (0, 1), 'title': 'B08'},
    'B8A': {'cmap': 'gray', 'scale': (0, 1), 'title': 'B8A'},
    'B09': {'cmap': 'gray', 'scale': (0, 1), 'title': 'B09'},
    'B10': {'cmap': 'gray', 'scale': (0, 1), 'title': 'B10'},
    'B11': {'cmap': 'gray', 'scale': (0, 1), 'title': 'B11'},
    'B12': {'cmap': 'gray', 'scale': (0, 1), 'title': 'B12'},
    'texture': {'cmap': 'inferno', 'scale': (0, 1), 'title': 'Texture'},
    'gabor': {'cmap': 'cubehelix', 'scale': (0, 1), 'title': 'Gabor'},
    'hog': {'cmap': 'cividis', 'scale': (0, 1), 'title': 'HOG'},
    'sobel': {'cmap': 'gray', 'scale': (0, 1), 'title': 'Sobel'},
    'canny': {'cmap': 'binary', 'scale': (0, 1), 'title': 'Canny'},
    'l': {'cmap': 'inferno', 'scale': (0, 1), 'title': 'Lightness'},
    'a': {'cmap': 'inferno', 'scale': (0, 1), 'title': '*a'},
    'b': {'cmap': 'inferno', 'scale': (0, 1), 'title': '*b'},
    'h': {'cmap': 'cubehelix', 'scale': (0, 1), 'title': 'Hue'},
    's': {'cmap': 'cubehelix', 'scale': (0, 1), 'title': 'Saturation'},
    'v': {'cmap': 'cubehelix', 'scale': (0, 1), 'title': 'Value'},
    'bgr': {'cmap': 'ocean_r', 'scale': (-1, 1), 'title': 'Blue/Green Ratio'},
    'calculate_pseudo_subsurface_depth': {'cmap': 'ocean_r', 'scale': (-1, 1), 'title': 'Pseudo Bathymetry'},
    "calculate_water_surface_index": {'cmap': 'ocean_r', 'scale': (-1, 1), 'title': 'Water Surface Index'},
    "ci": {'cmap': 'RdYlGn', 'scale': (-1, 1), 'title': 'Chlorophyll Index'},
    "ndci": {'cmap': 'RdYlGn', 'scale': (-1, 1), 'title': 'ND Chlorophyll Index'},
    "oci": {'cmap': 'RdYlGn', 'scale': (-1, 1), 'title': 'Ocean Colour Index'},
    "ssi": {'cmap': 'ocean_r', 'scale': (-1, 1), 'title': 'Sea Surface Index'},
    "ti": {'cmap': 'BrBG_r', 'scale': (-1, 1), 'title': 'Turbidity Index'},
    "wqi": {'cmap': 'BrBG_r', 'scale': (-1, 1), 'title': 'Water Quality Index'},
    'cl_oci': {'cmap': 'RdYlGn', 'scale': (-1, 1), 'title': 'Chlorophyll OCI'},
}

fig, axs = plt.subplots(grid_rows, grid_cols, figsize=(20, 15))  # Adjust the figsize as needed


for i, feature in enumerate(feature_names):
    # Calculate the 5th and 95th percentiles of the data for each feature
    min_val, max_val = np.percentile(feature_stack[:,:,i], [10, 90])

    # Assuming a grayscale colormap for spectral bands and RdYlGn for others
    cmap = 'gray' if feature.startswith('B') or feature in ['texture', 'gabor', 'hog', 'sobel', 'canny'] else 'RdYlGn'

    # Update the dictionary with the calculated scale and colormap
    feature_properties[feature] = {'cmap':  feature_properties[feature].get('cmap', feature), 'scale': (min_val, max_val), 'title': feature_properties[feature].get('title', feature)}


for i, ax in enumerate(axs.flatten()):
    if i < n_features:
        # Get feature properties
        feature_name = feature_names[i]
        properties = feature_properties.get(feature_name, {})
        cmap = properties.get('cmap', 'gray')
        scale = properties.get('scale', (0, 1))
        title = properties.get('title', feature_name)
        arrow = FancyArrowPatch((0.1, 0.1), (0.1, 0.2), transform=ax.transAxes, arrowstyle='->', color='white')
        ax.add_patch(arrow)
        ax.text(0.1, 0.21, 'N', transform=ax.transAxes, color='white', ha='center')
        # Display the image
        img = ax.imshow(feature_stack[:,:,i], cmap=cmap)
        ax.set_title(title)

        # Create a ScalarMappable and add colorbar
        norm = Normalize(vmin=scale[0], vmax=scale[1])
        mappable = ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array([])
        fig.colorbar(mappable, ax=ax, orientation='vertical')

        # Add scale bar
        scalebar = AnchoredSizeBar(ax.transData,
                                   100, '1 km', 'lower right', 
                                   pad=0.1,
                                   color='white',
                                   frameon=False,
                                   size_vertical=1)
        ax.add_artist(scalebar)

        # Hide axes ticks
        ax.set_xticks([])
        ax.set_yticks([])

    else:
        # Hide unused subplots
        ax.set_visible(False)

plt.tight_layout()
plt.show()