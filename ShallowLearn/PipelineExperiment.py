#import sklearn pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, PowerTransformer, MinMaxScaler

import numpy as np
import pandas as pd

import ShallowLearn.ImageHelper as ih
from ShallowLearn.IndiceFeatures import get_feature_order
from ShallowLearn.band_mapping import band_mapping


features = [i[0] for i in get_feature_order()]
features.append("mask")
features = list(band_mapping.keys()) + features

indices = np.load("/media/zba21/Expansion/Cleaned_Data_Directory/indices.npy")
masks = np.load("/media/zba21/Expansion/Cleaned_Data_Directory/masks.npy")
images = np.load("/media/zba21/Expansion/Cleaned_Data_Directory/imgs.npy")
masks = np.uint8(masks)

data_combined = np.concatenate((images, indices, masks),axis = 3)



# Define the columns for each transformer
power_transformer_columns = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12','calculate_water_surface_index']
standard_scaler_columns = ['bgr', 'ci', 'ndci', 'oci', 'ssi', 'ti', 'wqi']
minmax_scaler_columns = ['B01', 'calculate_pseudo_subsurface_depth']
passthrough_columns = ['mask']
# Create transformers
power_transformer_transformer = ('power_transformer', PowerTransformer(), power_transformer_columns)
standard_scaler_transformer = ('standard_scaler', StandardScaler(), standard_scaler_columns)
minmax_scaler_transformer = ('minmax_scaler', MinMaxScaler(), minmax_scaler_columns)
passthrough_transformer = ('passthrough', 'passthrough', passthrough_columns)
# Initialize ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        power_transformer_transformer,
        standard_scaler_transformer,
        minmax_scaler_transformer,
        passthrough_transformer
    ]
)

# # Initialize Pipeline
# pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

# data_expanded = data_combined.reshape(-1, len(features))
# df = pd.DataFrame(data_expanded, columns = features).dropna()
# transformed_data = pipeline.fit_transform(df)

df = pd.read_csv("Data/transformed_data.csv")
print(df.head())