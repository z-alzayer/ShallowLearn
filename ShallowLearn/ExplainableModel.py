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
import ShallowLearn.IndiceFeatures as indice_features

from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from umap import UMAP


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
feature_names = ['texture', 'gabor', 'hog', 'sobel', 'canny']
features = generate_features(ih.plot_rgb(median_images))
median_image = trf.LCE_multi(remove_zeros_from_image(median_images))
indice_object = indice_features.GenerateIndicesPerImage(median_images)

def create_stack(img):
    features = generate_features(ih.plot_rgb(img))
    indice_object = indice_features.GenerateIndicesPerImage(img)
    feature_stack = np.concatenate((img, features, indice_object.indices), axis = 2)
    return feature_stack

feature_stack = np.concatenate((median_image, features, indice_object.indices), axis = 2)

X = feature_stack.reshape(-1, feature_stack.shape[2])

# import pca and plot the explained variance ratio
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X = imputer.fit_transform(X)

# pca = PCA(n_components=3)
# pca.fit(X)
umap_model = UMAP(n_components=5)
umap_model.fit(X)
transformed = umap_model.transform(X)
# transformed = pca.transform(X)
plt.scatter(transformed[:, 0], transformed[:, 1], alpha=0.5)
plt.show()

# import kmeans and fuzzy cmeans and plot the elbow plot
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from skfuzzy.cluster import cmeans
from skfuzzy.cluster import cmeans_predict
from sklearn.metrics import calinski_harabasz_score
# import standard scaler
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

scores = []
X = transformed
# for i in range(3, 10):
kmeans = DBSCAN().fit(X)
labels = kmeans.labels_
# score = silhouette_score(X, labels, metric='euclidean')
calinski_score = calinski_harabasz_score(X, labels)
# print("For n_clusters = {}, silhouette score is {})".format(i, score))
print("calinski_harabasz_score is {})".format(calinski_score))
scores.append(calinski_score)
# plt.scatter(range(3, 10), scores)
# plt.xlabel('Number of Clusters')
# plt.ylabel('Calinski Harabasz Score')
# plt.show()

# optimal_k = np.argmax(scores) + 3
# print(f'Optimal k is {optimal_k}')
print(f'Number of clusters is {len(np.unique(labels))}')

# # Apply Fuzzy C-Means with the optimal number of clusters
# # cntr, u, u0, d, jm, p, fpc = cmeans(X.T, c=optimal_k, m=2, error=0.005, maxiter=1000)
# labels = np.argmax(u, axis=0)
# plt.figure(figsize=(8, 6))
# feature_1 = 2
# feature_2 = 8
# # for j in range(optimal_k):
#     # plt.scatter(X[labels == j, feature_1]/255, X[labels == j, feature_2]/255, label=f'Cluster {j+1}', alpha=0.5)

# # Plot the cluster centers
# plt.scatter(cntr[:, feature_1]/255, cntr[:, feature_2]/255, s=200, c='black', marker='x', label='Centroids')

# plt.title('Clusters and Centroids')
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.legend()
# plt.show()
print(labels.shape)
print(labels)
label_dict = {1:"Wave Front",2:"Sea",0:"Coral Reef"}
ih.discrete_implotv2(labels.reshape( 517, 271), string_labels=label_dict)
plt.show()