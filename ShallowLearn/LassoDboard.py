import numpy as np
from dash import Dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output

# Generate synthetic data (100x100 image with 3 spectral dimensions)
x, y = 100, 100
spectral_dims = 10
data = np.random.random((x, y, spectral_dims))

app = Dash(__name__)

app.layout = html.Div([
    dcc.Graph(
        id='image-graph',
        config={'staticPlot': False, 'displayModeBar': True},
        figure={
            'data': [{
                'z': data[:, :, 0],
                'type': 'heatmap',
                'colorscale': 'gray',
                'showscale': False,
            }],
            'layout': {
                'dragmode': 'lasso'
            }
        }
    ),
    dcc.Graph(id='spectral-graph'),
])

@app.callback(
    Output('spectral-graph', 'figure'),
    [Input('image-graph', 'selectedData')]
)
def update_spectral_graph(selectedData):
    # For testing, let's just generate random data for the spectral graph.
    # This is to see if the callback mechanism is working.
    spectra = np.random.random(spectral_dims)
    return {
        'data': [{'x': list(range(spectral_dims)), 'y': spectra}],
        'layout': {
            'title': 'Spectral Information',
            'xaxis': {'title': 'Spectral Dimension'},
            'yaxis': {'title': 'Intensity'}
        }
    }

if __name__ == '__main__':
    app.run_server(debug=True)
 