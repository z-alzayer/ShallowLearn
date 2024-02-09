from qgis.core import QgsProject, QgsRasterDataProvider, QgsPointXY, QgsCoordinateTransform, QgsCoordinateReferenceSystem
from osgeo import gdal, ogr, osr
import numpy as np
from sklearn.cluster import KMeans
from qgis.core import (QgsProject, QgsRasterLayer, QgsVectorLayer, QgsApplication, QgsCoordinateTransform, QgsCoordinateReferenceSystem, QgsRasterLayer)
import ShallowLearn.QgsInterface as slq

label_color_dict = {
    0: {'color': 'Red', 'label': 'Plataeu ', 'opacity': 50},
    1: {'color': 'Green', 'label': 'Reef Flat', 'opacity': 65},
    2: {'color': 'Blue', 'label': 'Cloud', 'opacity': 50},
    3: {'color': 'Orange', 'label': 'Sea / Dark Pixels', 'opacity': 60},
    4: {'color': 'Gray', 'label': 'Slope', 'opacity': 60},
    5: {'color': 'Cyan', 'label': 'Vegetation', 'opacity': 60},
    6: {'color': 'Pink', 'label': 'Bare Ground', 'opacity': 50},
    7: {'color': 'White', 'label': 'Category 3', 'opacity': 60},
}
rlayer = QgsProject.instance().mapLayersByName('S2B_MSIL1C_20220406T003659_N0400_R059_T55LCD_20220406T015555.SAFE')[0]  # Raster layer
vlayer = QgsProject.instance().mapLayersByName('K-Init')[0]  # Vector layer containing points
initial_centroids = slq.initiate_centroids(rlayer, vlayer)[:,[4,3,2,8]]
print(initial_centroids.shape)
output_path =  "/tmp/clipped_raster27.tif"
# Get raster extent and path
slq.clip_area(rlayer, output_path)

ds = gdal.Open(output_path)

X = slq.gdal_interface(output_path).gdal_to_np()
label_shape = X.shape[:2]
print(label_shape)
X = X[:,:,[4,3,2,8]].reshape(-1, 4) / 10_000

# Stack arrays along the third dimension and then reshape for scikit-learn
#X = np.stack(bands, axis=-1).reshape((-1, band_count)) / 10_000

# Ensure the number of centroids matches the expected number of clusters
n_clusters = min(len(initial_centroids), 10)  # Adjust based on your needs
k_means_model = KMeans(n_clusters=n_clusters, init=initial_centroids[:n_clusters], 
                        max_iter = 29, n_init=1).fit(X)
print(k_means_model.labels_)

# Load the clipped raster as a temporary layer
#clipped_layer = iface.addRasterLayer(output_path, "Clipped Raster")

# Reshape KMeans labels to the original raster shape
labels_reshaped = k_means_model.labels_.reshape(label_shape) + 1

slq.write_label_raster_out(labels_reshaped, ds, "kmeans_clf")
# Load the KMeans classification result as a new layer in QGIS
#kmeans_layer = iface.addRasterLayer('/tmp/kmeans_raster.tif', 'KMeans Classification')

# Path to the raster file
src_raster_path = '/tmp/kmeans_clf.tif'
# Output path for the polygonized vector
out_vector_path = '/tmp/kmeans_classification.shp'

slq.vectorize_labels(src_raster_path, "kmeans_clf", "kmeans classification")
slq.render_layering(label_color_dict, "kmeans classification")