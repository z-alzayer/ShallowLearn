from qgis.core import QgsProject, QgsRasterDataProvider, QgsPointXY, QgsCoordinateTransform, QgsCoordinateReferenceSystem
from osgeo import gdal, ogr, osr
import numpy as np
from sklearn.cluster import KMeans
from qgis.core import QgsProject, QgsRasterLayer, QgsVectorLayer, QgsApplication
from qgis.core import QgsProject, QgsCoordinateTransform, QgsCoordinateReferenceSystem
from sklearn.mixture import GaussianMixture
# Corrected import for QgsRasterLayer if needed
from qgis.core import QgsRasterLayer



label_color_dict = {
    0: 'Red',
    1: 'Green',
    2: 'Blue',
    3: 'Yellow',
    4: 'Magenta',
    5: 'Cyan',
    6: 'Orange',
    7: 'Grey'
}



def extract_raster_values_at_points(rlayer, vlayer):
    """
    Extract raster values at point locations specified in a vector layer.
    
    Parameters:
    - rlayer: The raster layer from which to extract values.
    - vlayer: The vector layer containing point geometries.
    
    Returns:
    - A list of dictionaries with point IDs and their corresponding raster values across all bands.
    """
    # Prepare the coordinate transformation from the vector layer's CRS to the raster layer's CRS
    source_crs = vlayer.crs()
    target_crs = rlayer.crs()
    transform = QgsCoordinateTransform(source_crs, target_crs, QgsProject.instance())
    
    # Get raster band count and geotransform
    band_count = rlayer.bandCount()
    geotransform = rlayer.dataProvider().sourceNoDataValue(1)
    
    # Initialize a list to hold raster values for each point
    raster_values = []
    
    # Iterate over each feature (point) in the vector layer
    for feature in vlayer.getFeatures():
        geom = feature.geometry()
        # Transform the geometry to the raster's CRS
        geom.transform(transform)
        point = QgsPointXY(geom.asPoint())
        
        # Extract the pixel values from the raster
        values = {}
        for band in range(1, band_count + 1):
            value, success = rlayer.dataProvider().sample(point, band)
            if success:
                values[f'Band_{band}'] = value
            else:
                values[f'Band_{band}'] = None
        
        # Append the extracted values for this point to the list
        raster_values.append({'PointID': feature.id(), 'Values': values})
    
    return raster_values
    
rlayer = QgsProject.instance().mapLayersByName('S2A_MSIL1C_20160820T003712_N0204_R059_T55LCD_20160820T003707.SAFE')[0]  # Raster layer
vlayer = QgsProject.instance().mapLayersByName('Test')[0]  # Vector layer containing points

# Extract raster values
extracted_values = extract_raster_values_at_points(rlayer, vlayer)
print(extracted_values)


# Extract the initial centroids from the vector layer points
initial_centroids_values = extract_raster_values_at_points(rlayer, vlayer)
# Process the extracted values to fit KMeans centroids initialization
initial_centroids = np.array([list(centroid['Values'].values()) for centroid in initial_centroids_values if None not in centroid['Values'].values()])

# Get raster extent and path
extent = iface.mapCanvas().extent()
xmin, xmax, ymin, ymax = extent.toRectF().left(), extent.toRectF().right(), extent.toRectF().bottom(), extent.toRectF().top()
output_path = "/tmp/clipped_raster.tif"
rlayer_path = rlayer.dataProvider().dataSourceUri().split("|")[0]

# Clip the raster
gdal.Warp(output_path, rlayer_path, outputBounds=[xmin, ymin, xmax, ymax], dstSRS=rlayer.crs().toWkt())

# Open the clipped raster and convert to array
ds = gdal.Open(output_path)
band_count = ds.RasterCount
bands = [ds.GetRasterBand(i+1).ReadAsArray() for i in range(band_count)]

# Stack arrays along the third dimension and then reshape for scikit-learn
X = np.stack(bands, axis=-1).reshape((-1, band_count))

# Ensure the number of centroids matches the expected number of clusters
n_clusters = min(len(initial_centroids), 8)  # Adjust based on your needs
k_means_model = GaussianMixture(n_components=n_clusters,
                                     n_init=1).fit(X)
print(k_means_model.predict(X))

# Load the clipped raster as a temporary layer
clipped_layer = iface.addRasterLayer(output_path, "Clipped Raster")

# Reshape KMeans labels to the original raster shape
n_rows, n_cols = bands[0].shape
labels_reshaped = k_means_model.predict(X).reshape((n_rows, n_cols))

# Create a new raster for the labels
driver = gdal.GetDriverByName('GTiff')
out_raster = driver.Create('/tmp/kmeans_raster.tif', n_cols, n_rows, 1, gdal.GDT_Float32)
out_band = out_raster.GetRasterBand(1)
out_band.WriteArray(labels_reshaped)

# Copy the geotransform and projection from the original raster to the new one
out_raster.SetGeoTransform(ds.GetGeoTransform())
out_raster.SetProjection(ds.GetProjection())

# Flush and close the output raster
out_band.FlushCache()
out_raster = None
ds = None

# Load the KMeans classification result as a new layer in QGIS
kmeans_layer = iface.addRasterLayer('/tmp/kmeans_raster.tif', 'KMeans Classification')

# Path to the raster file
src_raster_path = '/tmp/kmeans_raster.tif'
# Output path for the polygonized vector
out_vector_path = '/tmp/kmeans_classification.shp'

src_ds = gdal.Open(src_raster_path)
src_band = src_ds.GetRasterBand(1)
drv = ogr.GetDriverByName('ESRI Shapefile')
out_ds = drv.CreateDataSource(out_vector_path)
out_layer = out_ds.CreateLayer('', srs=None)

# Add a field to the layer
new_field = ogr.FieldDefn('DN', ogr.OFTInteger)
out_layer.CreateField(new_field)

# Polygonize
gdal.Polygonize(src_band, None, out_layer, 0, [], callback=None)

out_ds = None  # Save and close the shapefile
kmeans_vector_layer = iface.addVectorLayer(out_vector_path, 'KMeans Classification', 'ogr')



# Create a list to hold our categories
categories = []

for label, color_name in label_color_dict.items():
    # Initialize the symbol for this category
    symbol = QgsSymbol.defaultSymbol(kmeans_vector_layer.geometryType())
    symbol.setColor(QColor(color_name))
    
    # Create the category and add it to the list
    category = QgsRendererCategory(label, symbol, str(label))
    categories.append(category)

# Create the renderer and assign it to the layer
renderer = QgsCategorizedSymbolRenderer('DN', categories)
kmeans_vector_layer.setRenderer(renderer)

# Refresh the layer's symbology
kmeans_vector_layer.triggerRepaint()
iface.layerTreeView().refreshLayerSymbology(kmeans_vector_layer.id())