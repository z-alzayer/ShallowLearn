import ShallowLearn.ImageHelper as ih
import numpy as np
from sklearn.decomposition import PCA
import ShallowLearn.FileProcessing as fp
from sklearn.preprocessing import StandardScaler, RobustScaler
import cv2
import matplotlib.pyplot as plt 
import ShallowLearn.Graphing as gp
import ShallowLearn.ClusterManipulation as cm
from ShallowLearn.Indices import calculate_water_surface_index
import pandas as pd


def pad_image(img, max_width, max_height):
    height, width, channels = img.shape
    
    top_pad = (max_height - height) // 2
    bottom_pad = max_height - height - top_pad
    left_pad = (max_width - width) // 2
    right_pad = max_width - width - left_pad
    
    padded_channels = []
    for c in range(channels):
        channel = img[:, :, c]
        padded_channel = cv2.copyMakeBorder(channel, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=0)
        padded_channels.append(padded_channel)
    
    return np.stack(padded_channels, axis=-1)

def preprocess_images(images):
    global max_width, max_height
    max_width, max_height = 0, 0
    
    # Find the maximum dimensions
    for img in images:
        h, w, _ = img.shape
        max_width = max(max_width, w)
        max_height = max(max_height, h)
    
    # Pad images
    padded_images = [pad_image(img, max_width, max_height) for img in images]
    return np.array([img.reshape(-1) for img in padded_images])  # Flatten images

def plot_scree(pca):
    explained_variance_ratio = pca.explained_variance_ratio_
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(explained_variance_ratio)), explained_variance_ratio)
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Scree Plot')
    plt.savefig('scree_plot.png')
    plt.show()

def plot_reconstructed_images(images, pca, components=[1, 10, 50]):
    mean = np.mean(images, axis=0)
    # mean = mean.reshape(1, -1)
    print(mean.shape)
    for n in components:
        pca.n_components = n
        transformed = pca.transform(images)
        reconstructed = pca.inverse_transform(transformed) 
        reconstructed = reconstructed.reshape((-1, max_height, max_width, 13))  # Adjust shape as per your images
        print(reconstructed.shape)
        plt.imshow(ih.plot_rgb(reconstructed[0]))  # Adjust index to view different images
        plt.title(f'Reconstructed Image with {n} Components')
        plt.axis('off')
        plt.savefig(f'Graphs/Timelapse_PCA/reconstructed_image_{n}_components.png')
        plt.show()




path = "/mnt/sda_mount/Clipped/L1C/"
files = fp.list_files_in_dir_recur(path)
# print(files)
files = [i for i in files if "/44" in i and i.endswith(".tiff")]
# print(len(files))
# files = fp.filter_files(files, 1, 10)

# Assume images is a list of your loaded images as numpy arrays
images = [ih.load_img(path) for path in files]

# Preprocess images
# images = preprocess_images(images)
original_shape = np.array(images).shape
images = np.array(images).reshape(70, -1)
print(images.shape)
# max_width, max_height = 220,223
# Standardize features
scaler = StandardScaler()
images_standardized = scaler.fit_transform(images)

# Apply PCA
pca = PCA(n_components=2)  # You can change the number of components
pca_result = pca.fit_transform(images_standardized)
# plt.plot(pca_result[:, 0], pca_result[:, 1], 'o')
# plt.savefig("Graphs/Timelapse_PCA/PCA.png")


# # Print explained variance ratio
# print("Explained Variance Ratios:", pca.explained_variance_ratio_)

# # Save scree plot
# plot_scree(pca)

# # Save reconstructed images (optional)
# plot_reconstructed_images(images, pca)
colors = ['red', 'green', 'blue']

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42).fit(pca_result[:, :2])
labels = kmeans.labels_
label_colors = [colors[label] for label in labels]
cluster_centers = kmeans.cluster_centers_
# plt.scatter(pca_result[:, 0], pca_result[:, 1], c=label_colors) 
# plt.colorbar()  # Optionally add a colorbar to show the mapping of colors to values
# plt.title('PCA Clustered')
# plt.savefig("Graphs/Timelapse_PCA/PCA_clustered.png")
# images = np.array([ih.plot_rgb(i) for i in images.reshape(original_shape)])

# gp.plot_images_on_PCA(pca_result, images, labels, threshold=50, title = "Satellite Image Visualization in Principal Component Space", save = True, path = "Graphs/Timelapse_PCA/", ext = '.png')

images = images.reshape(original_shape)

leftmost_cluster_index, min_x = cm.find_leftmost_cluster(cluster_centers)

print(min_x, leftmost_cluster_index)
good_images = []
file_names = []
wsi = []


for i,image_name in zip(labels, files):
    if i == leftmost_cluster_index:
        print(i)
        good_images.append(images[i])
        # good_images = np.array(good_images)
        file_names.append(image_name)
        wsi.append(calculate_water_surface_index(images[i]))
print(len(good_images))

filtered_imagery = pd.DataFrame(file_names)
print(filtered_imagery.head())
filtered_imagery.to_csv("Data/filtered_imagery_reef_44.csv")