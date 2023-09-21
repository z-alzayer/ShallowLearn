import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import widgets
from matplotlib import colors

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
from PIL import Image
import io
import ShallowLearn.ImageHelper as ih
from ShallowLearn.Training import create_row_data_frame
from sklearn.decomposition import PCA
import base64


# original_data = np.load("/media/ziad/Expansion/Cleaned_Data_Directory/radiometrically_normalized.npy")
# pixel_vectors = []
# image_indices = []

# for idx, img in enumerate(original_data):
#     h, w, _ = img.shape
#     for i in range(h):
#         for j in range(w):
#             pixel_vectors.append(img[i, j])
#             image_indices.append(idx)

# pixel_vectors = np.array(pixel_vectors)
# pca = PCA(n_components=2)
# transformed_pixel_data = pca.fit_transform(pixel_vectors)


# sample_size = 10000

# # Randomly sample indices
# random_indices = np.random.choice(len(pixel_vectors), sample_size, replace=False)

# # Subsample the data and image indices
# subsampled_pixel_vectors = pixel_vectors[random_indices]
# subsampled_image_indices = [image_indices[i] for i in random_indices]
# pca = PCA(n_components=2)
# transformed_subsampled_pixel_data = pca.fit_transform(subsampled_pixel_vectors)
# app = dash.Dash(__name__)

# # Create a simple scatter plot using plotly express
# fig = px.scatter(x=transformed_subsampled_pixel_data[:, 0], y=transformed_subsampled_pixel_data[:, 1])

# app.layout = html.Div([
#     dcc.Graph(id='pca-plot', figure=fig),
#     html.Img(id='hover-data-image', src='', width=200)  # Adjust width as necessary
# ])

# @app.callback(
#     Output('hover-data-image', 'src'),
#     [Input('pca-plot', 'hoverData')]
# )
# def update_image(hoverData):
#     if hoverData is None:
#         return dash.no_update  # No update if no data
    
#     # Get the index of the point
#     point_idx = hoverData['points'][0]['pointIndex']
    
#     # Get the index of the source image using the subsampled_image_indices list
#     source_image_idx = subsampled_image_indices[point_idx]
    
#     # Get the corresponding source image
#     img_array = original_data[source_image_idx]
#     img = Image.fromarray(ih.plot_rgb(img_array))
    
#     buffered = io.BytesIO()
#     img.save(buffered, format="PNG")
#     img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
#     return f"data:image/png;base64,{img_str}"

# if __name__ == '__main__':
#     app.run_server(debug=True)




### Image visualisation 2 pcas


# original_data = np.load("/media/ziad/Expansion/Cleaned_Data_Directory/radiometrically_normalized.npy")

# pca = PCA(n_components=2)
# transformed_data = pca.fit_transform(original_data.reshape(original_data.shape[0], -1))

# app = dash.Dash(__name__)

# # Create a simple scatter plot using plotly express
# fig = px.scatter(x=transformed_data[:, 0], y=transformed_data[:, 1])

# app.layout = html.Div([
#     dcc.Graph(id='pca-plot', figure=fig),
#     html.Img(id='hover-data-image', src='', width=200)  # Adjust width as necessary
# ])

# @app.callback(
#     Output('hover-data-image', 'src'),
#     [Input('pca-plot', 'hoverData')]
# )
# def update_image(hoverData):
#     if hoverData is None:
#         return dash.no_update  # No update if no data
    
#     # Get the index of the point
#     point_idx = hoverData['points'][0]['pointIndex']
    
#     # Get the corresponding image
#     img_array = original_data[point_idx]
#     img = Image.fromarray(ih.plot_rgb(img_array))
    
#     buffered = io.BytesIO()
#     img.save(buffered, format="PNG")
#     img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
#     return f"data:image/png;base64,{img_str}"

# if __name__ == '__main__':
#     app.run_server(debug=True)
