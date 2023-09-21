import pandas as pd 
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import ShallowLearn.Transform as tf
import math 
import os 
import joblib
import matplotlib.pyplot as plt
from ShallowLearn.Training import create_dataframe, PATH, create_row_data_frame

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
import umap
from sklearn.model_selection import ParameterGrid

PATH = "/media/ziad/Expansion/Cleaned_Data_Directory/"
graph_path = "/home/ziad/Documents/Github/ShallowLearn/Graphs"
SAMPLE_SIZE = 200_000

param_grid = {
    'pca_comp': [2, 5, 10],
    'umap_neigh': [45, 300],
    'umap_metric': ["canberra", "correlation"]
}


def create_scatter_cmap(df, cols = None, alpha = None):
    """converts colour space to a format compatible with matplotlib scatter
        of values between -1 and 1 then linearly converts it back to 0 and 1
          - using B08 as alpha, input columns are
        a list"""
    # make a copy of the dataframe and so that changes are not inplace
    df_copy = df.copy()

    if cols is not None and len(cols) > 4:
        raise ValueError("Too many columns, only 4 allowed")

    if cols is None:
        # We assume the alpha is the last channel
        cols = ["B04", "B03", "B02", "B08"]

    for column in cols:
        df_copy[column] = np.clip(df_copy[column], -1, 1)
    for column in cols:
        df_copy[column] = (df_copy[column] + 1) / 2

    if alpha is not None and len(cols) == 4:
        df_copy[cols[-1]] = df_copy[cols[-1]] * alpha

    df_copy['colour'] = list(zip(*[df_copy[col] for col in cols]))
    return df_copy['colour']

def reshape_to_square(image, N = 4):
    # Calculate the total number of elements divided by the channel size (4 in this case)
    total_elements = image.size // image.shape[-1]
    
    # Find n
    n = math.ceil(math.sqrt(total_elements))
    
    # Create a zero-filled reshaped_image
    reshaped_image = np.zeros((n, n, N))
    
    # Flatten the original image for easier indexing
    flat_image = image.reshape(-1, image.shape[-1])
    
    # Fill the reshaped_image
    for idx, row in enumerate(flat_image):
        i = idx // n
        j = idx % n
        reshaped_image[i, j] = row
        
    return reshaped_image


def sort_multidimensional_data(image):
    """
    Sorts a multidimensional array based on the average value of the last dimension.

    Parameters:
    - image (ndarray): A multidimensional numpy array.

    Returns:
    - ndarray: The image sorted by the average value of the last dimension.
    """

    # Compute the 'sorting criterion' as the mean along the last axis
    # Simple sorting method - if we sort by all the arrays its too ram intensive
    sorting_criterion = np.mean(image, axis=-1)

    # Get sorting indices from sorting criterion
    sorting_indices = np.argsort(sorting_criterion.ravel())

    # Flatten and sort the original image with these indices
    sorted_flattened = image.reshape(-1, image.shape[-1])[sorting_indices]

    # Reshape to get the sorted image
    sorted_data = sorted_flattened.reshape(image.shape)
    
    return sorted_data

def load_training_raw(path = None):
    """Local dataloader for the data in the Cleaned_Data_Directory
        - currently just for testing """
    if path is not None:
        indices = np.load(os.path.join(path, "indices.npy"))
        images = np.load(os.path.join(path, "imgs.npy"))
        data_combined = np.concatenate((images, indices), axis = 3)
    return data_combined

def create_pipeline(pca_comp, umap_neigh, umap_metric):
    return Pipeline([
        ('pca', PCA(n_components=pca_comp, random_state=42)),
        ('umap', umap.UMAP(n_components=3, n_jobs=-1, n_neighbors=umap_neigh, metric=umap_metric, random_state=42))
    ])

def save_plot(embedding, plot_type, pca_comp, umap_neigh, umap_metric):
    if plot_type == "2D":
        plt.scatter(embedding[:, 0], embedding[:, 1], c=create_scatter_cmap(df, alpha=0.1))
    elif plot_type == "3D":
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], c=create_scatter_cmap(df, alpha=0.1))
    
    filename = f"{plot_type}_pca{pca_comp}_umap_neigh{umap_neigh}_metric{umap_metric}_{SAMPLE_SIZE}.png"
    plt.savefig(os.path.join(graph_path, filename))
    plt.close()


# pipeline_pca_umap = Pipeline([
#     ('pca', PCA(n_components=15, random_state=42)),  # reduce dimensionality before t-SNE
#     ('umap', umap.UMAP(n_components=3, n_jobs= -1, n_neighbors= 35, metric = "canberra", random_state=42))  # you usually want 2D or 3D for visualization
# ])


# pipeline_pca = Pipeline([
#     ('pca', PCA(n_components=15, random_state=42))  # reduce dimensionality before t-SNE
# ])






if __name__ == "__main__":

    df = create_dataframe(PATH)
    df = df.loc[df['mask'] == 1]
    print("dropped mask values")
    df = df.drop(columns = ['mask'])
    pipe = joblib.load("/home/ziad/Documents/Github/ShallowLearn/Models/preproc_pipeline.pkl")
    sample = df.sample(SAMPLE_SIZE, random_state = 42)
    df = pd.DataFrame(pipe.transform(sample), columns = pipe.feature_names_in_)
    for params in ParameterGrid(param_grid):
        print(params)
        pipeline = create_pipeline(**params)
        embedding = pipeline.fit_transform(df)
        save_plot(embedding, "2D", **params)
        save_plot(embedding, "3D", **params)

    
    # print("transforming to umap embedding")
    # embedding = pipeline_pca.fit_transform(df)
    # plt.scatter(embedding[:, 0], embedding[:, 1], c = create_scatter_cmap(df, alpha = 0.1))
    # plt.show()

    # #3d scatter 
    # from mpl_toolkits.mplot3d import Axes3D
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # #plot embedding in 3d
    # ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], c = create_scatter_cmap(df, alpha = 0.1))
    # plt.show()