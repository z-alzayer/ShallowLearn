import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import numpy as np
from sklearn.decomposition import PCA

# Assuming `images` is your list or array of images and `pca_results` is your PCA output
# For demonstration, we will create some random data
images = np.random.randint(0, 255, (10, 10, 10, 3), dtype=np.uint8)  # 10 images, 10x10 pixels, 3 channels
pca_results = np.random.rand(10, 2)  # Random PCA results for 10 points

app = dash.Dash(__name__)

# Layout of your dash app:
app.layout = html.Div([
    dcc.Graph(
        id='pca-scatter',
        figure={
            'data': [
                go.Scatter(
                    x=pca_results[:, 0],
                    y=pca_results[:, 1],
                    mode='markers',
                    marker={'size': 10}
                )
            ],
            'layout': go.Layout(
                title='PCA Results',
                clickmode='event+select'
            )
        }
    ),
    html.Div(id='image-display')
])

# Callback for updating display image on click
@app.callback(
    Output('image-display', 'children'),
    [Input('pca-scatter', 'clickData')]
)
def display_image(clickData):
    if clickData:
        point_index = clickData['points'][0]['pointIndex']
        img = images[point_index]
        # Convert the image array into a format that can be displayed by html.Img:
        img_string = np_to_base64(img)
        return html.Img(src='data:image/png;base64,{}'.format(img_string))
    return "Click on a point in the PCA plot to display the corresponding image."

# Function to convert image array into a base64 string
def np_to_base64(img):
    from PIL import Image
    import base64
    from io import BytesIO
    
    im = Image.fromarray(img)
    with BytesIO() as buffer:
        im.save(buffer, 'png')
        return base64.b64encode(buffer.getvalue()).decode()

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8888)