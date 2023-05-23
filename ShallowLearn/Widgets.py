import matplotlib.pyplot as plt
import matplotlib.colors as colors
import ipywidgets as widgets
import numpy as np
from IPython.display import display

class BandPlotter:
    def __init__(self, image, cmap='gray', discrete_cmap=False, figsize = (15, 10)):
        self.image = image
        self.band = 0
        self.cmap = plt.get_cmap(cmap)
        self.discrete_cmap = discrete_cmap

        # Ensure the image has at least one channel
        assert len(self.image.shape) == 3, "Image should be 3-dimensional (height, width, channels)"

        # Create the figure and axis
        self.fig, self.ax = plt.subplots(1, 2, figsize=figsize)
        self.img_plot = self.ax[0].imshow(self.image[:,:,self.band], cmap=self.get_cmap())
        self.cbar = self.fig.colorbar(self.img_plot, ax=self.ax[0], orientation='vertical')
        self.plot_histogram()

        # Create the slider
        self.slider = widgets.IntSlider(min=0, max=self.image.shape[2]-1, step=1, value=self.band, description='Band')
        self.slider.observe(self.update_band, names='value')

        display(self.slider)

    def update_plot(self):
        self.ax[0].clear()
        self.img_plot = self.ax[0].imshow(self.image[:,:,self.band], cmap=self.get_cmap())
        self.cbar.mappable.set_clim(vmin=np.min(self.image[:,:,self.band]), vmax=np.max(self.image[:,:,self.band]))
        self.cbar.update_normal(self.img_plot)
        self.plot_histogram()
        self.fig.canvas.draw()

    def update_band(self, change):
        self.band = change.new
        self.update_plot()

    def plot_histogram(self, bins=50, min_value=1):
        self.ax[1].clear()
        channel_data = self.image[:,:,self.band].flatten()
        channel_data = channel_data[channel_data >= min_value]
        histogram, bins = np.histogram(channel_data, bins=bins, range=(0, np.max(self.image)))
        bin_centers = 0.5*(bins[1:] + bins[:-1])

        # Plot the histogram using line plot
        self.ax[1].plot(bin_centers, histogram, label=f'Band {self.band + 1}')

        # Customize the plot
        self.ax[1].set_xlabel('Value')
        self.ax[1].set_ylabel('Frequency')
        self.ax[1].set_title('Histogram of Each Band')
        self.ax[1].legend()
        
    def get_cmap(self):
        if self.discrete_cmap:
            # Create a discrete colormap
            max_val = np.max(self.image)
            boundaries = np.arange(max_val + 2) - 0.5
            self.cmap = colors.ListedColormap(self.cmap(np.arange(max_val + 1)))
            self.cmap.set_over(self.cmap(max_val))
            self.cmap.set_under('black')
            self.cmap = colors.BoundaryNorm(boundaries, self.cmap.N, clip=True)
        return self.cmap
