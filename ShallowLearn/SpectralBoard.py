import numpy as np
import os
from dash import Dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go

import ShallowLearn.ImageHelper as ih
from ShallowLearn.Misc import sentinel_2_satellite_data
import ShallowLearn.FileProcessing as fp


import ShallowLearn.FileProcessing as fp
path = "/mnt/sda_mount/Clipped/L1C/"
images = fp.list_files_in_dir_recur(path)
single_reef = [i for i in images if "34_" in i]
# len(single_reef)

# # Generate synthetic data (100x100 image with 3 spectral dimensions)
# x, y = 100, 100
# spectral_dims = 10

# Make a synthetic image for testing later
# data = np.random.random((x, y, spectral_dims))

# path = "/mnt/sda_mount/Clipped/L1C/S2A_MSIL1C_20160323T003752_N0201_R059_T55LCD_20160323T003830.SAFE"
# images = os.listdir(path)
images = single_reef

average_wavelengths = np.mean([sentinel_2_satellite_data['Sentinel-2A Wavelength'], sentinel_2_satellite_data['Sentinel-2B Wavelength']], axis=0)
average_bandwidths = np.mean([sentinel_2_satellite_data['Bandwidth A'], sentinel_2_satellite_data['Bandwidth B']], axis=0)

app = Dash(__name__)

app.layout = html.Div([
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='Image Explorer', value='tab-1'),
        dcc.Tab(label='Another Dashboard', value='tab-2'),
    ]),
    html.Div(id='tabs-content')
])

@app.callback(Output('tabs-content', 'children'),
              [Input('tabs', 'value')])
def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            dcc.Dropdown(
                id='image-dropdown',
                options=[{'label': img, 'value': img} for img in images],
                value=images[0],
                clearable=False
            ),
            dcc.Graph(
                id='image-rgb'
            ),
            dcc.Graph(id='spectral-graph')
        ])
    elif tab == 'tab-2':
        return html.Div([
            html.H3('Content for another dashboard')
            # Add your components for the second dashboard here
        ])

@app.callback(
    [Output('image-rgb', 'figure'),
     Output('spectral-graph', 'figure')],
    [Input('image-dropdown', 'value'),
     Input('image-rgb', 'clickData'),
     Input('image-rgb', 'relayoutData')]
)

def update_display(selected_image, clickData, relayoutData):
    data = ih.load_img(os.path.join(path, selected_image))
    rgb_image = ih.plot_rgb(data)
    rgb_image = (rgb_image - np.min(rgb_image)) / (np.max(rgb_image) - np.min(rgb_image))
    fig = px.imshow(rgb_image)
    fig.update_layout(clickmode='event+select')
    
    # Preserve zoom and pan
    if relayoutData and 'xaxis.range[0]' in relayoutData:
        fig['layout']['xaxis'].update(range=[relayoutData['xaxis.range[0]'], relayoutData['xaxis.range[1]']])
        fig['layout']['yaxis'].update(range=[relayoutData['yaxis.range[0]'], relayoutData['yaxis.range[1]']])

    # Handle click data for spectral graph
    if not clickData:
        return fig, {}
    
    x_coord = int(clickData['points'][0]['x'])
    y_coord = int(clickData['points'][0]['y'])
    spectra = data[y_coord, x_coord]

    spectral_fig = go.Figure()

    # Adding a line connecting the midpoints
    spectral_fig.add_trace(go.Scatter(
        x=average_wavelengths,
        y=spectra,
        mode='lines',
        name='Spectral Signature',
        line=dict(color='blue')
    ))

    for i, wavelength in enumerate(average_wavelengths):
        spectral_fig.add_trace(go.Scatter(
            x=[wavelength],
            y=[spectra[i]],
            mode='markers',
            name=sentinel_2_satellite_data['Band Name'][i],
            error_x=dict(
                type='data',
                array=[average_bandwidths[i] / 2],
                visible=True
            )
        ))

    spectral_fig.update_layout(
        title=f'Spectral Information for Point ({x_coord}, {y_coord})',
        xaxis_title='Wavelength (nm)',
        yaxis_title='Digital Number (DN)',
        xaxis=dict(tickvals=average_wavelengths, ticktext=sentinel_2_satellite_data['Band Name'])
    )

    return fig, spectral_fig

if __name__ == '__main__':
    app.run_server(debug=True)

