import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
import numpy as np

import ShallowLearn.ImageHelper as ih

path = "/mnt/sda_mount/Clipped/L2A/S2A_MSIL2A_20180502T003711_N9999_R059_T55LCD_20231001T134316.SAFE/44_S2A_MSIL2A_20180502T003711_N9999_R059_T55LCD_20231001T134316.SAFE.tiff"

# Generate synthetic multichannel (13 bands) image of size (100, 100)
synthetic_image = ih.load_img(path)

# Select 3 random bands to display as an RGB image
bands_to_display = [4,3,2]
rgb_image = synthetic_image[:, :, bands_to_display]

# Choose two bands for the scatter plot
band_x, band_y = np.random.choice(synthetic_image.shape[2], 2, replace=False)
scatter_x = synthetic_image[:, :, band_x].flatten()
scatter_y = synthetic_image[:, :, band_y].flatten()

# Keep the original indices of the flattened image to reference back to the image data
indices = np.arange(synthetic_image.shape[0] * synthetic_image.shape[1])

# Plot the synthetic image and scatter plot
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
image_plot = axs[0].imshow(rgb_image, aspect='auto')
scatter_plot = axs[1].scatter(scatter_x, scatter_y, s=5)

class LassoTool:
    def __init__(self, ax, scatter_plot, image_plot, synthetic_image, indices):
        self.ax = ax
        self.scatter_plot = scatter_plot
        self.image_plot = image_plot
        self.synthetic_image = synthetic_image
        self.indices = indices
        self.canvas = ax.figure.canvas
        self.rgb_image = rgb_image
        self.lasso = LassoSelector(ax, onselect=self.on_select)
        self.selected_indices = []

    def on_select(self, verts):
        path = Path(verts)
        ind = np.nonzero([path.contains_point(xy) for xy in self.scatter_plot.get_offsets()])[0]
        self.selected_indices = self.indices[ind]
        self.highlight_points(ind)
        self.highlight_image(ind)

    def highlight_image(self, ind):
        # Create a mask for the 3 channels of the RGB image initialized to False
        mask = np.zeros((self.synthetic_image.shape[0], self.synthetic_image.shape[1]), dtype=bool)
        
        # Convert the selected indices from the scatter plot to indices in the RGB image
        image_indices = np.unravel_index(self.indices[ind], mask.shape)
        
        # Update the mask to True for the selected points
        mask[image_indices] = True
        
        # Create a highlighted version of the RGB image
        highlighted_image = self.rgb_image.copy()
        # Suppose we want to highlight with a red color
        highlighted_image[..., 0][mask] = 255  # Red channel
        highlighted_image[..., 1][mask] = 0    # Green channel
        highlighted_image[..., 2][mask] = 0    # Blue channel
        
        # Update the image plot
        self.image_plot.set_data(highlighted_image)
        self.canvas.draw_idle()


    def highlight_points(self, ind):
        # Ensure we have the same number of facecolors as scatter points
        facecolors = self.scatter_plot.get_facecolor()
        if len(facecolors) != len(self.scatter_plot.get_offsets()):
            # Initialize facecolors to transparent
            facecolors = np.ones((len(self.scatter_plot.get_offsets()), 4)) * [0, 0, 0, 0.1]
        
        # Reset all points to low visibility
        facecolors[:, -1] = 0.1
        
        # Highlight the selected points
        facecolors[ind, :3] = [1, 0, 0]  # Red color for selected points
        facecolors[ind, -1] = 1  # Full alpha for selected points
        
        # Set the updated colors back to the scatter plot
        self.scatter_plot.set_facecolor(facecolors)
        
        self.canvas.draw_idle()

        # Highlight corresponding points in the image
        mask = np.zeros(self.synthetic_image.shape[:2], dtype=bool)
        mask[self.indices[ind] // self.synthetic_image.shape[1], self.indices[ind] % self.synthetic_image.shape[1]] = True
        
        # Overlay the mask on the image_plot
        highlighted_image = np.copy(self.synthetic_image[:, :, bands_to_display])
        for channel in range(3):  # Assuming RGB
            # Skip the channel if it's not one of the displayed bands
            if channel in bands_to_display:
                highlighted_image[..., channel] = np.where(mask, 1, highlighted_image[..., channel])
        self.image_plot.set_data(highlighted_image)
        self.canvas.draw_idle()
# Initialize the lasso tool
lasso_tool = LassoTool(axs[1], scatter_plot, image_plot, synthetic_image, indices)

plt.show()
