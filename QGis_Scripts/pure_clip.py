from qgis.core import QgsProject, QgsRasterDataProvider, QgsPointXY, QgsCoordinateTransform, QgsCoordinateReferenceSystem
from osgeo import gdal, ogr, osr
import numpy as np
from sklearn.cluster import KMeans
from qgis.core import (QgsProject, QgsRasterLayer, QgsVectorLayer, QgsApplication, QgsCoordinateTransform, QgsCoordinateReferenceSystem, QgsRasterLayer)
import ShallowLearn.QgsInterface as slq

output_path = "/tmp/test_area.tiff"
rlayer = QgsProject.instance().mapLayersByName('S2B_MSIL1C_20220406T003659_N0400_R059_T55LCD_20220406T015555.SAFE')[0]  # Raster layer
slq.clip_area(rlayer, output_path)