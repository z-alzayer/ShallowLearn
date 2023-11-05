import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from skimage.transform import resize
import os
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def plot_images_on_PCA(transformed_data, original_images, labels, threshold=50, title = "Satellite Image Visualization in Principal Component Space", save = False, path = None, ext = '.png'):
    """
    Overlay resized images onto the PCA plot if they are far apart.
    :param transformed_data: PCA transformed data.
    :param original_images: The original images.
    :param labels: Labels or colors for each point.
    :param threshold: Distance threshold to consider points as far apart.
    """
    distances = squareform(pdist(transformed_data))
    plt.figure(figsize=(15, 15))
    
    plt.scatter(transformed_data[:, 0], transformed_data[:, 1], c=labels, cmap='rainbow')
    plt.title(title, fontsize = 20)
    plt.xlabel('Principal Component 1', fontsize=18)
    plt.ylabel('Principal Component 2',  fontsize=18)
    plt.ticklabel_format(style='plain', axis='x', useOffset=False)
    plt.tick_params(axis='both', which='major', labelsize=16)
    for i in range(len(transformed_data)):
        for j in range(i+1, len(transformed_data)):
            if distances[i, j] > threshold:
                img = original_images[i]  # you might need to adjust the image processing here
                im = OffsetImage(img, zoom=0.2)  # Adjust zoom here
                ab = AnnotationBbox(im, (transformed_data[i, 0], transformed_data[i, 1]), 
                                    box_alignment=(0.5, 0.5), frameon=False)
                plt.gca().add_artist(ab)
    if save == True:
        plt.savefig(os.path.join(path , title) + ext)
    plt.show()