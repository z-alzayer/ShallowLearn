import ShallowLearn.QgsInterface as slq
import ShallowLearn.Models as models
import numpy as np

from osgeo import gdal
labels_layer = "kmeans_classification"

original_array_path = slq.get_file_path('clipped_raster27')
print("---------------")
print(original_array_path)
extent = slq.get_raster_extent(original_array_path)

rlayer = QgsProject.instance().mapLayersByName('kmeans classification')[0]  # vector layer
rlayer_path = rlayer.dataProvider().dataSourceUri()
rasterize_label = slq.rasterize_vector_to_match_raster(rlayer_path, "/tmp/test.tiff"
                                                        ,original_array_path)
                                           
y = slq.gdal_interface("/tmp/test.tiff").gdal_to_np() -1
y_shape = y.shape
y = y.reshape(-1)
X = slq.gdal_interface(original_array_path).gdal_to_np()
X = X.reshape(-1, 13) / 10_000
log_reg = models.logistic_regression_pipeline(X,y)

test = slq.gdal_interface("/tmp/test_area.tiff").gdal_to_np() -1
test_shape = test.shape
test = test.reshape(-1, 13) / 10_000

prediction = log_reg.predict(test)
test_object = slq.gdal_interface("/tmp/test_area.tiff")
print(prediction.shape)
# import matplotlib.pyplot as plt 
# plt.imshow(np.flip(prediction.reshape(test_shape[:2])))
# plt.show()
print(prediction.reshape(test_shape[:2]).shape)
slq.write_label_raster_out(prediction.reshape(test_shape[:2]), test_object.image, "log_reg_pred")

