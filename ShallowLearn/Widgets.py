import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import widgets
from matplotlib import colors

class BandPlotter:
    def __init__(self, image, cmap='gray', discrete_cmap=False, figsize=(15, 10), band_mapping=None):
        self.image = image
        self.cmap = plt.get_cmap(cmap)
        self.discrete_cmap = discrete_cmap
        self.band_mapping = band_mapping
        self.band_keys = list(band_mapping.keys()) if band_mapping else list(range(image.shape[2]))
        self.band = self.band_keys[0]

        # Ensure the image has at least one channel
        assert len(self.image.shape) == 3, "Image should be 3-dimensional (height, width, channels)"

        # Create the figure and axis
        self.fig, self.ax = plt.subplots(1, 2, figsize=figsize)
        self.img_plot = self.ax[0].imshow(self.image[:,:,self.get_band_index()], cmap=self.get_cmap())
        self.cbar = self.fig.colorbar(self.img_plot, ax=self.ax[0], orientation='vertical')
        self.plot_histogram()

        # Create the slider
        self.slider = widgets.IntSlider(min=0, max=len(self.band_keys)-1, step=1, value=0, description='Band')
        self.slider.observe(self.update_band, names='value')

        display(self.slider)
    
    def get_band_index(self):
        if self.band_mapping:
            return self.band_mapping[self.band]['index']
        else:
            return self.band

    def update_plot(self):
        self.ax[0].clear()
        self.img_plot = self.ax[0].imshow(self.image[:,:,self.get_band_index()], cmap=self.get_cmap())
        self.cbar.mappable.set_clim(vmin=np.min(self.image[:,:,self.get_band_index()]), vmax=np.max(self.image[:,:,self.get_band_index()]))
        self.cbar.update_normal(self.img_plot)
        self.plot_histogram()
        self.fig.canvas.draw()

    def update_band(self, change):
        self.band = self.band_keys[change.new]
        self.update_plot()

    def plot_histogram(self, bins=50, min_value=0):
        self.ax[1].clear()
        channel_data = self.image[:,:,self.get_band_index()].flatten()
        valid_data = channel_data[channel_data >= min_value]

        if valid_data.size == 0:
            print(f"Skipping histogram for band {self.band}: No data values above {min_value}")
            return

        histogram, bins = np.histogram(valid_data, bins=bins, range=(np.min(valid_data), np.max(valid_data)))
        bin_centers = 0.5*(bins[1:] + bins[:-1])

        if self.band_mapping:
            self.ax[1].plot(bin_centers, histogram, label=f"{self.band_mapping[self.band]['name']} (Band {self.band_mapping[self.band]['index'] + 1})")
        else:
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