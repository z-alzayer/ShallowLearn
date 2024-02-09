from osgeo import gdal, ogr, osr
import numpy as np

from qgis.core import (QgsProject, QgsRendererCategory, QgsCategorizedSymbolRenderer, 
                       QgsSymbol, QgsCoordinateReferenceSystem, QgsPointXY, QgsCoordinateTransform)
from qgis.utils import iface
from PyQt5.QtGui import QColor


def get_file_path(layer_name):
    """Returns a filepath from a qgis project layer name"""
    layer = QgsProject.instance().mapLayersByName(layer_name)[0]
    return layer.dataProvider().dataSourceUri()


def load_in_gdal(layer_name):
    """Loads a layer into gdal from a qgis project layer name"""
    filepath = get_file_path(layer_name)
    return gdal.Open(filepath)

def get_interface_extent():
    # iface = QgisInterface()
    canvas = iface.mapCanvas()
    extent = canvas.extent()
    return extent

def clip_area(rlayer, output_path, layer_name = "Clipped Layer2",  show = True):
    qrectangle = get_interface_extent()
    xmin, xmax, ymin, ymax= qrectangle.toRectF().left(), qrectangle.toRectF().right(), qrectangle.toRectF().bottom(), qrectangle.toRectF().top()
    rlayer_path = get_file_path(rlayer.name())
    result = gdal.Warp(output_path, rlayer_path, outputBounds=[xmin, ymin, xmax, ymax], dstSRS=rlayer.crs().toWkt())
    return (result)
    # if show:
    #     iface.addRasterLayer(result, layer_name)


def write_label_raster_out(labels_reshaped, original_ds, out_name, show = True):
    """write a raster out to temp directory
        labels_reshaped: labels array (n_rows, n_cols)
        out_name: str
        - works on mac / linux for now
          adds to temp and loads in qgis"""

    driver = gdal.GetDriverByName('GTiff')
    out_raster = driver.Create(f'/tmp/{out_name}.tif', labels_reshaped.shape[1], labels_reshaped.shape[0], 1, gdal.GDT_Float32)
    out_band = out_raster.GetRasterBand(1)
    print(out_band)
    out_band.WriteArray(labels_reshaped)

    # Copy the geotransform and projection from the original raster to the new one
    out_raster.SetGeoTransform(original_ds.GetGeoTransform())
    out_raster.SetProjection(original_ds.GetProjection())

    # Flush and close the output raster
    out_band.FlushCache()
    out_raster = None
    original_ds = None

    # Load the KMeans classification result as a new layer in QGIS
    #kmeans_layer = iface.addRasterLayer('/tmp/kmeans_raster.tif', 'KMeans Classification')

    # Path to the raster file
    src_raster_path = f'/tmp/{out_name}.tif'
    if show:
        iface.addRasterLayer(src_raster_path, out_name)


def vectorize_labels(label_raster_path, out_path, qgis_name="Vectorised Layer", field_name="DN", epsg_code=None):
    """Converts a raster to a vector, optionally setting a specific EPSG for the output vector."""

    out_vector_path = f'/tmp/{out_path}.shp'

    src_ds = gdal.Open(label_raster_path)
    src_band = src_ds.GetRasterBand(1)

    # Retrieve the raster's spatial reference
    raster_srs = osr.SpatialReference()
    raster_srs.ImportFromWkt(src_ds.GetProjection())

    # If an EPSG code is provided, override the raster's spatial reference with the new one
    if epsg_code:
        new_srs = osr.SpatialReference()
        new_srs.ImportFromEPSG(epsg_code)
    else:
        new_srs = raster_srs  # Use the raster's original spatial reference

    drv = ogr.GetDriverByName('ESRI Shapefile')
    out_ds = drv.CreateDataSource(out_vector_path)
    out_layer = out_ds.CreateLayer('', srs=new_srs)  # Set the spatial reference for the output layer

    # Add a field to the layer
    new_field = ogr.FieldDefn(field_name, ogr.OFTInteger)
    out_layer.CreateField(new_field)

    # Polygonize
    gdal.Polygonize(src_band, None, out_layer, 0, [], callback=None)

    out_ds = None  # Save and close the shapefile

    # Assuming this function is run within QGIS with access to iface
    vector_layer = iface.addVectorLayer(out_vector_path, qgis_name, 'ogr')

    return vector_layer
def get_raster_extent(file_path):
    """
    Get the spatial extent of a raster file.
    
    Parameters:
    - file_path: Path to the raster file.
    
    Returns:
    - A tuple representing the spatial extent (minX, maxX, minY, maxY).
    """
    # Open the raster file
    ds = gdal.Open(file_path)
    if ds is None:
        raise Exception("Could not open the raster file at {}".format(file_path))

    # Get the geotransformation and dimensions
    gt = ds.GetGeoTransform()
    x_size = ds.RasterXSize
    y_size = ds.RasterYSize

    # Calculate the extent
    minX = gt[0]
    maxY = gt[3]
    maxX = minX + (gt[1] * x_size)
    minY = maxY + (gt[5] * y_size)

    return (minX, maxX, minY, maxY)



def rasterize_vector_to_match_raster(vector_path, output_raster_path, reference_raster_path, attribute_name="DN", pixel_size=10.0):
    """
    Rasterize a vector file to match the size and resolution of a reference raster.

    Parameters:
    - vector_path: Path to the input vector file.
    - output_raster_path: Path where the output raster will be saved.
    - reference_raster_path: Path to the reference raster from which dimensions and geo-transform are derived.
    - attribute_name: Name of the attribute field in the vector file to use for rasterization values.
    - pixel_size: The resolution of the pixels in the output raster, assumed to match the reference raster.
    """
    
    # Open the reference raster to get its dimensions and geotransformation
    ref_ds = gdal.Open(reference_raster_path)
    if ref_ds is None:
        raise Exception(f"Could not open reference raster: {reference_raster_path}")
    
    x_res = ref_ds.RasterXSize
    y_res = ref_ds.RasterYSize
    geo_transform = ref_ds.GetGeoTransform()
    
    # Create the output raster with the same dimensions and geo-transform as the reference
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(output_raster_path, x_res, y_res, 1, gdal.GDT_Byte)
    if out_ds is None:
        raise Exception("Could not create the output raster file.")
    out_ds.SetGeoTransform(geo_transform)
    out_ds.SetProjection(ref_ds.GetProjection())
    
    # Open the vector file
    vector_ds = ogr.Open(vector_path)
    if vector_ds is None:
        raise Exception(f"Could not open vector file: {vector_path}")
    vector_layer = vector_ds.GetLayer()
    
    # Perform rasterization using the specified attribute for values
    gdal.RasterizeLayer(out_ds, [1], vector_layer, options=[f"ATTRIBUTE={attribute_name}"])
    
    # Cleanup
    out_ds = None
    vector_ds = None
    ref_ds = None

    print(f"Rasterization completed. Output saved to: {output_raster_path}")
def render_layering(label_color_dict, vector_layer_name, field_name = "DN"):
    

# Create a list to hold our categories
    vector_layer = QgsProject.instance().mapLayersByName(vector_layer_name)[0]
    categories = []
    print(vector_layer)

    for label, info in label_color_dict.items():
        color_name = info['color']
        category_label = info['label']
        opacity = info['opacity']

        # Initialize the symbol for this category
        symbol = QgsSymbol.defaultSymbol(vector_layer.geometryType())
        symbol.setColor(QColor(color_name))
        symbol.setOpacity(opacity / 100.0)  # QgsSymbol expects opacity as a fraction
        
        # Create the category and add it to the list
        category = QgsRendererCategory(label, symbol, category_label)
        categories.append(category)

    # Create the renderer and assign it to the layer
    renderer = QgsCategorizedSymbolRenderer(field_name, categories)
    vector_layer.setRenderer(renderer)

    # Refresh the layer's symbology
    vector_layer.triggerRepaint()
    iface.layerTreeView().refreshLayerSymbology(vector_layer.id())


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
                values[f'Band_{band}'] = value / 10_000
            else:
                values[f'Band_{band}'] = None
        
        # Append the extracted values for this point to the list
        raster_values.append({'PointID': feature.id(), 'Values': values})
    
    return raster_values

def initiate_centroids(rlayer, vlayer):
    initial_centroids_values = extract_raster_values_at_points(rlayer, vlayer)
# Process the extracted values to fit KMeans centroids initialization
    initial_centroids = np.array([list(centroid['Values'].values()) for centroid in initial_centroids_values if None not in centroid['Values'].values()])
    return initial_centroids


class gdal_interface():

    def __init__(self, file_path):
        self.file_path = file_path
        self.image = gdal.Open(file_path)
        self.band_count = self.image.RasterCount

    def gdal_to_np(self):
        bands = [self.image.GetRasterBand(i).ReadAsArray() for i in range(1, self.band_count + 1)]
        return np.dstack(bands)
