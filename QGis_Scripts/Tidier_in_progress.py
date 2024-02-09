from qgis.core import (QgsProject, QgsRasterLayer, QgsVectorLayer, QgsCoordinateTransform,
                       QgsCoordinateReferenceSystem, QgsSymbol, QgsRendererCategory, QgsCategorizedSymbolRenderer)
from qgis.PyQt.QtGui import QColor
from osgeo import gdal, ogr
import numpy as np
from sklearn.cluster import KMeans

# Global variables for the label-color mapping
label_color_dict = {
    0: {'color': 'Red', 'label': 'Plateau', 'opacity': 50},
    1: {'color': 'Green', 'label': 'Reef Flat', 'opacity': 65},
    2: {'color': 'Blue', 'label': 'Cloud', 'opacity': 50},
    3: {'color': 'Orange', 'label': 'Sea / Dark Pixels', 'opacity': 60},
    4: {'color': 'Gray', 'label': 'Slope', 'opacity': 60},
    5: {'color': 'Cyan', 'label': 'Vegetation', 'opacity': 60},
    6: {'color': 'Pink', 'label': 'Bare Ground', 'opacity': 50},
    7: {'color': 'White', 'label': 'Category 3', 'opacity': 60},
}

def extract_raster_values_at_points(rlayer, vlayer):
    source_crs = vlayer.crs()
    target_crs = rlayer.crs()
    transform = QgsCoordinateTransform(source_crs, target_crs, QgsProject.instance())
    
    band_count = rlayer.bandCount()
    raster_values = []
    
    for feature in vlayer.getFeatures():
        geom = feature.geometry()
        geom.transform(transform)
        point = QgsPointXY(geom.asPoint())
        
        values = {}
        for band in range(1, band_count + 1):
            value, success = rlayer.dataProvider().sample(point, band)
            if success:
                values[f'Band_{band}'] = value / 10_000  # Assuming normalization
            else:
                values[f'Band_{band}'] = None
        
        raster_values.append({'PointID': feature.id(), 'Values': values})
    
    return raster_values

def k_means_classification(rlayer, vlayer, clip_path, n_clusters=8, max_iter=300):
    extracted_values = extract_raster_values_at_points(rlayer, vlayer)
    initial_centroids = np.array([list(centroid['Values'].values()) for centroid in extracted_values if None not in centroid['Values'].values()])
    print(initial_centroids)

    ds = gdal.Open(rlayer.dataProvider().dataSourceUri().split("|")[0])
    band_count = ds.RasterCount
    bands = [ds.GetRasterBand(i+1).ReadAsArray() for i in range(band_count)]
    X = np.stack(bands, axis=-1).reshape((-1, band_count)) / 10_000  # Normalization
    
    k_means_model = KMeans(n_clusters=min(len(initial_centroids), n_clusters), init=initial_centroids[:n_clusters], max_iter=max_iter, n_init=1).fit(X)
    
    return k_means_model.labels_.reshape(bands[0].shape)
# def k_means_classification(processed_array, centroids = None,v_layer = None,r_layer = None, n_clusters=8, max_iter=300):
#     """
#     Applies KMeans classification to a processed raster array.

#     Args:
#         processed_array: The NumPy array of the raster data.
#         n_clusters: The number of clusters for KMeans.
#         max_iter: The maximum number of iterations for KMeans.

#     Returns:
#         The reshaped labels from the KMeans classification.
#     """
#     if centroids is not None and vlayer is not None and v_layer is not None:
#         extracted_values = extract_raster_values_at_points(rlayer, vlayer)
#         initial_centroids = np.array([list(centroid['Values'].values()) for centroid in extracted_values if None not in centroid['Values'].values()])
#         k_means_model = KMeans(n_clusters=n_clusters, max_iter=max_iter, n_init=1, init=initial_centroids[:n_clusters]).fit(processed_array)
#         labels_reshaped = k_means_model.labels_.reshape((-1, processed_array.shape[2]))
#     else:
#         k_means_model = KMeans(n_clusters=n_clusters, max_iter=max_iter, n_init=1).fit(processed_array)
#         labels_reshaped = k_means_model.labels_.reshape((-1, processed_array.shape[2]))
    
#     return labels_reshaped
    
    
def polygonize_raster_to_shapefile(src_raster_path, out_vector_path):
    src_ds = gdal.Open(src_raster_path)
    src_band = src_ds.GetRasterBand(1)
    drv = ogr.GetDriverByName('ESRI Shapefile')
    out_ds = drv.CreateDataSource(out_vector_path)
    out_layer = out_ds.CreateLayer('', srs=None)
    
    new_field = ogr.FieldDefn('DN', ogr.OFTInteger)
    out_layer.CreateField(new_field)
    
    gdal.Polygonize(src_band, None, out_layer, 0, [], callback=None)
    out_ds = None  # Save and close the shapefile

def apply_styling_to_vector_layer(vector_layer, label_color_dict):
    categories = []
    
    for label, info in label_color_dict.items():
        color_name = info['color']
        category_label = info['label']
        opacity = info['opacity']
        
        symbol = QgsSymbol.defaultSymbol(vector_layer.geometryType())
        symbol.setColor(QColor(color_name))
        symbol.setOpacity(opacity / 100.0)
        
        category = QgsRendererCategory(label, symbol, category_label)
        categories.append(category)
    
    renderer = QgsCategorizedSymbolRenderer('DN', categories)
    vector_layer.setRenderer(renderer)
    
    vector_layer.triggerRepaint()
    iface.layerTreeView().refreshLayerSymbology(vector_layer.id())

def clip_raster_to_extent(rlayer, output_path):
    """
    Clips the raster layer to the current map canvas extent.

    Args:
        rlayer: The raster layer to be clipped.
        output_path: Path to save the clipped raster.

    Returns:
        The path to the clipped raster.
    """
    extent = iface.mapCanvas().extent()
    xmin, ymin, xmax, ymax = extent.toRectF().getCoords()
    
    # Prepare gdal.Warp options
    options = gdal.WarpOptions(format='GTiff', outputBounds=[xmin, ymin, xmax, ymax], dstSRS=rlayer.crs().toWkt())
    gdal.Warp(output_path, rlayer.dataProvider().dataSourceUri().split("|")[0], options=options)

    return output_path



# Example of how to use the functions
def main():
    rlayer_name = 'S2B_MSIL1C_20220406T003659_N0400_R059_T55LCD_20220406T015555.SAFE'
    vlayer_name = 'K-Init'
    
    rlayer = QgsProject.instance().mapLayersByName(rlayer_name)[0]
    vlayer = QgsProject.instance().mapLayersByName(vlayer_name)[0]
    
    clipped_raster_path = '/tmp/clipped_raster.tif'
    clip_raster_to_extent(rlayer, clipped_raster_path)

    
    # Assuming you have a function to read the clipped raster and convert it to an array
    #processed_array = read_and_process_raster(clipped_raster_path)  # Placeholder for actual function
    
    labels_reshaped = k_means_classification(rlayer,vlayer,clipped_raster_path, n_clusters=8, max_iter=10)
    # # Assuming the steps to create '/tmp/kmeans_raster.tif' are executed here as before
    
    # src_raster_path = '/tmp/kmeans_raster.tif'
    # out_vector_path = '/tmp/kmeans_classification.shp'
    # polygonize_raster_to_shapefile(src_raster_path, out_vector_path)
    
    # kmeans_vector_layer = iface.addVectorLayer(out_vector_path, 'KMeans Classification', 'ogr')
    # apply_styling_to_vector_layer(kmeans_vector_layer, label_color_dict)

# Run the main function
main()
