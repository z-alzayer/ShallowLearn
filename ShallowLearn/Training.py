import pandas as pd
import numpy as np
import rasterio as rio 
import os
import pkg_resources


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn.mixture import GaussianMixture
from openTSNE.sklearn import TSNE

from ShallowLearn import LoadData
from ShallowLearn.band_mapping import band_mapping




def preprocess_data(data):
    loaded_data = LoadData.LoadFromCSV(data)
    specific_reef = loaded_data.load_specific_reef(6880)
    return np.array(specific_reef)

def reshape_data(data):
    # shape = data.shape
    dim_0 = data.shape[0] * data.shape[2] * data.shape[3] 
    channels = data.shape[1]
    return data.reshape(dim_0, channels)

class TrainOnFullReefs():

    def __init__(self):
        path = pkg_resources.resource_filename('ShallowLearn', '../Data/Clear_Reefs.csv')
        X_train = reshape_data(preprocess_data(path))
        print(X_train.shape)
        X_train = pd.DataFrame(X_train, columns = band_mapping.keys())
        # X_train = X_train.drop(columns = ["B10"], axis = 1)
        X_train = X_train.loc[~(X_train==0).any(axis=1)]
        X_train = X_train.drop_duplicates()
        print(X_train.shape)
        # Define the pipeline steps
        scaler = StandardScaler()
        imputer = SimpleImputer(strategy='mean')

        # Define the pipeline
        my_pipeline = Pipeline([
            ('scaler', scaler),
            ('pca', PCA(n_components=2, random_state=42)),
            ('kmeans', KMeans(n_clusters=10, random_state=42))
        ])

        # Define the hyperparameters to tune
        param_grid = {
            'pca__n_components': [2, 5, 10],
            'kmeans__n_clusters': [2, 3, 5, 7],
            'kmeans__init': ['k-means++', 'random']
        }

        # # Instantiate the GridSearchCV object
        # grid_search = GridSearchCV(my_pipeline, param_grid, cv=5)

        # # Fit the pipeline to the training data
        # grid_search.fit(X_train)

        # # # Get the best pipeline and its parameters
        # best_pipeline = grid_search.best_estimator_
        # best_params = grid_search.best_params_

        # print(best_pipeline)
        # print(best_params)

        # # Predict cluster labels for the test data using the best pipeline
        # labels = best_pipeline.predict(X_test)

        # # Compute the clustering score for the test data using the best pipeline
        # score = best_pipeline.score(X_test)
        my_pipeline.fit(X_train)
        import joblib
        joblib.dump(my_pipeline, '/home/ziad/Documents/Github/ShallowLearn/Models/pipeline_pca2_kmeans10.pkl')
    
    class TrainOnLABSpace():

        def __init__(self):
            path = pkg_resources.resource_filename('ShallowLearn', '../Data/Clear_Reefs.csv')
            X_train = reshape_data(preprocess_data(path))
            print(X_train.shape)
            X_train = np.unique(X_train, axis=0) # Remove duplicate rows
            print(X_train.shape)
            # Define the pipeline steps
            scaler = StandardScaler()
            imputer = SimpleImputer(strategy='mean')
if __name__ == "__main__":
    TrainOnFullReefs()
    # TrainOnFullReefs.TrainOnLABSpace()
