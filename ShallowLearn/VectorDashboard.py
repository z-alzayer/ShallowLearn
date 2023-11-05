import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import numpy as np
from sklearn.decomposition import PCA
from PIL import Image
import io
import base64
import ShallowLearn.ImageHelper as ih
from PIL import ImageDraw

# Sample data loading
original_data = np.load("/media/zba21/Expansion/Cleaned_Data_Directory/imgs.npy")

# Preparing pixel vectors and image indices
pixel_vectors = []
image_indices = []

for idx, img in enumerate(original_data):
    h, w, _ = img.shape
    for i in range(h):
        for j in range(w):
            pixel_vectors.append(img[i, j])
            image_indices.append(idx)

pixel_vectors = np.array(pixel_vectors)
image_indices = np.array(image_indices)  # Convert to numpy array for easy indexing

# Identify and remove black pixels
black_pixel_mask = np.all(pixel_vectors == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], axis=1)
valid_pixel_mask = ~black_pixel_mask

filtered_pixel_vectors = pixel_vectors[valid_pixel_mask]
filtered_image_indices = image_indices[valid_pixel_mask]  # This will now correctly filter out the corresponding image indices

# Subsampling
sample_size = 10000
random_indices = np.random.choice(len(filtered_pixel_vectors), sample_size, replace=False)
subsampled_pixel_vectors = filtered_pixel_vectors[random_indices]
subsampled_image_indices = filtered_image_indices[random_indices]
# PCA
pca = PCA(n_components=2)
transformed_subsampled_pixel_data = pca.fit_transform(subsampled_pixel_vectors)


def create_scatter_cmap_np(pixel_vectors, alpha=None):
    # Clip values to be between -1 and 1
    clipped_pixel_vectors = np.clip(pixel_vectors, -1, 1)
    
    # Normalize to range [0, 1]
    normalized_pixel_vectors = (clipped_pixel_vectors + 1) / 2

    # Apply alpha to the last channel, if given
    if alpha is not None:
        normalized_pixel_vectors[:, -1] *= alpha
    
    return normalized_pixel_vectors

# Apply the function on pixel_vectors
colours_array = create_scatter_cmap_np(subsampled_pixel_vectors)



# Color by Original Values
normalized_colors =colours_array[:, [3, 2, 1]]
print(np.max(normalized_colors, axis=0))
print(np.min(normalized_colors, axis=0))
# Setting up Dash




app = dash.Dash(__name__)

# Initial figure setup
fig = {
    'data': [
        {
            'x': transformed_subsampled_pixel_data[:, 0],
            'y': transformed_subsampled_pixel_data[:, 1],
            'mode': 'markers',
            'marker': {
                'color': normalized_colors.tolist(),
                'size': 5,
                'sizemode': 'diameter',
                'line': {
                    'width': 1,
                    'color': 'gray'
                }
            },
            'text': subsampled_image_indices,
            'hoverinfo': 'text'
        },
        {
            'x': [None],  # Empty trace for the highlighted point
            'y': [None],
            'mode': 'markers',
            'marker': {
                'color': 'black',
                'size': 10,
            },
            'name': 'Highlighted Pixel'
        }
    ]
}

app.layout = html.Div([
    dcc.Graph(id='pca-plot', figure=fig),
    html.Img(id='hover-data-image', src='', width=200)
])


def highlight_pixel(img, x, y, size=5, color=(255, 0, 0)):
    """Draw a square around the specific pixel."""
    draw = ImageDraw.Draw(img)
    draw.rectangle(
        [x - size, y - size, x + size, y + size],
        outline=color,
        width=2
    )
    return img

@app.callback(
    [Output('hover-data-image', 'src'),
     Output('pca-plot', 'figure')],
    [Input('pca-plot', 'hoverData')]
)
def update_image_and_highlight(hoverData):
    if hoverData is None:
        return dash.no_update, dash.no_update
    
    # Get the index of the point
    point_idx = hoverData['points'][0]['pointIndex']

    # Get the index of the source image using the subsampled_image_indices list
    source_image_idx = subsampled_image_indices[point_idx]
    
    # Determine the number of pixels in a single image
    num_pixels_single_image = original_data[0].shape[0] * original_data[0].shape[1]

    # Use modulo operation to get the pixel position within the specific image
    relative_pixel_position = random_indices[point_idx] % num_pixels_single_image

    pixel_y, pixel_x, _ = np.unravel_index(relative_pixel_position, original_data[source_image_idx].shape)
    
    # Get the corresponding source image
    img_array = original_data[source_image_idx].copy()
    img = Image.fromarray(ih.plot_rgb(img_array))
    
    # Highlight the specific pixel
    img = highlight_pixel(img, pixel_x, pixel_y)
    
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # Update the figure to include the highlighted point
    highlighted_x = transformed_subsampled_pixel_data[point_idx, 0]
    highlighted_y = transformed_subsampled_pixel_data[point_idx, 1]
    fig['data'][1]['x'] = [highlighted_x]
    fig['data'][1]['y'] = [highlighted_y]
    
    return f"data:image/png;base64,{img_str}", fig

if __name__ == '__main__':
    app.run_server(debug=True)
