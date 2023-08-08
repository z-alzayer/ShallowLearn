import pandas as pd
import numpy as np
import rasterio as rio 
import os
import pkg_resources


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn.mixture import GaussianMixture
from openTSNE.sklearn import TSNE
from sklearn.compose import ColumnTransformer


from ShallowLearn import LoadData
from ShallowLearn.band_mapping import band_mapping
from ShallowLearn.IndiceFeatures import get_feature_order


PATH = "/media/ziad/Expansion/Cleaned_Data_Directory/"

def preprocess_data(data):
    loaded_data = LoadData.LoadFromCSV(data)
    specific_reef = loaded_data.load_specific_reef(6880)
    return np.array(specific_reef)

def reshape_data(data):
    # shape = data.shape
    dim_0 = data.shape[0] * data.shape[2] * data.shape[3] 
    channels = data.shape[1]
    return data.reshape(dim_0, channels)


def create_dataframe(path = None):
    """Local dataloader class for the data in the Cleaned_Data_Directory"""
    features = [i[0] for i in get_feature_order()]
    features.append("mask")
    features = list(band_mapping.keys()) + features
    if path is not None:
        indices = np.load(os.path.join(path, "indices.npy"))
        masks = np.load(os.path.join(path, "masks.npy"))
        images = np.load(os.path.join(path, "imgs.npy"))
        masks = np.uint8(masks)    
        data_combined = np.concatenate((images, indices, masks), axis = 3)
    data_expanded = data_combined.reshape(-1, len(features))
    df = pd.DataFrame(data_expanded, columns = features).dropna()
    return df

class TrainPreprocess():
    # Just basic class for preprocessing data
    def __init__(self):
        power_transformer_columns = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12','calculate_water_surface_index']
        standard_scaler_columns = ['bgr', 'ci', 'ndci', 'oci', 'ssi', 'ti', 'wqi']
        minmax_scaler_columns = ['B01', 'calculate_pseudo_subsurface_depth']
        # passthrough_columns = ['mask']
        # Create transformers
        power_transformer_transformer = ('power_transformer', PowerTransformer(), power_transformer_columns)
        standard_scaler_transformer = ('standard_scaler', StandardScaler(), standard_scaler_columns)
        minmax_scaler_transformer = ('minmax_scaler', MinMaxScaler(), minmax_scaler_columns)
        # passthrough_transformer = ('passthrough', 'passthrough', passthrough_columns)
        # Initialize ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                power_transformer_transformer,
                standard_scaler_transformer,
                minmax_scaler_transformer
                # passthrough_transformer
            ]
        )

        # Initialize Pipeline
        print("initializing pipeline")
        pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
        df = create_dataframe(PATH)
        print("created dataframe")
        # drop values in df where mask is 0
        df = df.loc[df['mask'] == 1]
        print("dropped mask values")
        # drop mask column
        df = df.drop(columns = ['mask'])
        print("dropped mask column")
        # Fit the pipeline to the training data
        pipeline.fit(df)
        print("fit pipeline")
        import joblib
        # Save pipeline
        print("saving pipeline")
        joblib.dump(pipeline, '/home/ziad/Documents/Github/ShallowLearn/Models/preproc_pipeline.pkl')



class TrainOnFullReefs():

    def __init__(self):
        path = pkg_resources.resource_filename('ShallowLearn', '../Data/Clear_Reefs.csv')
        X_train = reshape_data(preprocess_data(path))
        print(X_train.shape)
        X_train = pd.DataFrame(X_train, columns = band_mapping.keys())
        X_train = X_train.drop(columns = ["B10"], axis = 1)
        X_train = X_train.loc[~(X_train==0).any(axis=1)]
        X_train = X_train.drop_duplicates()
        print(X_train.shape)
        # Define the pipeline steps
        scaler = PowerTransformer()
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
        joblib.dump(my_pipeline, '/home/ziad/Documents/Github/ShallowLearn/Models/pipeline_pca2_kmeans10pow_nb10.pkl')
    
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
    # TrainOnFullReefs()
    # TrainOnFullReefs.TrainOnLABSpace()
    TrainPreprocess()