import mpl_scatter_density # adds projection='scatter_density'
from matplotlib.colors import LinearSegmentedColormap
from mpl_scatter_density import ScatterDensityArtist
import seaborn as sns
from matplotlib import cm
from matplotlib.colors import Normalize 
from scipy.interpolate import interpn
import matplotlib.pyplot as plt
import numpy as np

white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
(0, '#ffffff'),
(1e-20, '#440053'),
(0.2, '#404388'),
(0.4, '#2a788e'),
(0.6, '#21a784'),
(0.8, '#78d151'),
(1, '#fde624'),
], N=256)

def set_limits(ax, x_scale = None, y_scale = None):
    if x_scale is not None:
        ax.set_xlim(x_scale)
    if y_scale is not None:
        ax.set_ylim(y_scale)


def density_scatter( x , y, ax, sort = True, bins = 20, **kwargs )   :
    """
    Scatter plot colored by 2d histogram
    """
    fig = ax.figure
    data , x_e, y_e = np.histogram2d( x, y, bins = bins, density = True )
    z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False)

    #To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    # Sort the points by density, so that the densest points are plotted last
    if sort :
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    ax.scatter( x, y, c=z, **kwargs )

    norm = Normalize(vmin = np.min(z), vmax = np.max(z))
    cbar = fig.colorbar(cm.ScalarMappable(norm = norm), ax=ax)
    cbar.ax.set_ylabel('Density')

    return ax

def kde_plot(ax, x, y,  x_scale = None, y_scale = None, cmap='coolwarm'):
    # Create scatter density plot using seaborn
    sns.kdeplot(x=x, y=y, ax=ax, cmap=cmap, fill=True)
    # ax.scatter(x, y, color='gray', alpha=0.1)  # scatter plot overlay
    set_limits(ax, x_scale, y_scale)

def hist_2d_plot(ax, x, y,  x_scale = None, y_scale = None, bins=30, cmap=white_viridis):
    # Create 2D histogram
    h = ax.hist2d(x, y, bins=bins, cmap=cmap)
    ax.figure.colorbar(h[3], ax=ax, label='Counts in bin')
    set_limits(ax, x_scale, y_scale)