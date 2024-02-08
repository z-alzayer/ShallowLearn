from ShallowLearn.IndiceFeatures import GenerateIndicesPerImage
from ShallowLearn import ImageHelper as ih
import ShallowLearn.ComputerVisionFeatures as cvf
from ShallowLearn.band_mapping import band_mapping
import numpy as np



def generate_features(image):
    # becareful with what you are passing in here
    if image.shape[-1] > 3:
        image = ih.plot_rgb(image)

    texture = cvf.texture_features(image, P=8, R=30)
    gabor = cvf.gabor_features(image, frequency = 2.5)
    hog = cvf.histogram_of_oriented_gradients(image)
    sobel = cvf.sobel_edge_detection(image)
    canny = cvf.edge_density(image)
    arr = np.array((texture, gabor, hog, sobel, canny))
    arr = np.swapaxes(arr, 0, 1)
    arr = np.swapaxes(arr, 1, 2)
    return arr

def column_names(indice_object):
    # remove hardcoded feature names if adding more features
    feature_names = ['texture', 'gabor', 'hog', 'sobel', 'canny']
    columns = list(band_mapping.keys()) + feature_names + indice_object.indice_order
    return columns



def create_stack(img):

    indice_generator = GenerateIndicesPerImage(img)
    features = generate_features(ih.plot_rgb(img))
    feature_stack = np.concatenate((img, features, indice_generator.indices), axis = 2)
    cols = column_names(indice_generator)

    return feature_stack, cols