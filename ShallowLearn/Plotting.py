import matplotlib.pyplot as plt 
import numpy as np
from matplotlib.animation import PillowWriter
import matplotlib.animation as animation

def animate_images_and_timeseries(images, df, x=50, y=50,fps=2, filename='animation.gif'):
    """
    Creates and saves a timed animation showing images with a red dot and arrow on the left,
    and scatter plots of three bands against dates on the right.
    
    Parameters:
    - images: list of np.array
        A list of images to be displayed.
    - df: pandas.DataFrame
        A DataFrame containing the time series data with datetime indices.
    - x: int
        The x-coordinate where the dot and arrow will be placed in the images.
    - y: int
        The y-coordinate where the dot and arrow will be placed in the images.
    - filename: str
        The name of the file where the animation will be saved.
    """
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    def update(frame):
        axs[0].cla()
        axs[1].cla()
        
        # Displaying the image with a red dot and arrow
        img = images[frame]
        axs[0].imshow(img)
        axs[0].plot(x, y, 'ro')
        axs[0].arrow(x, y-30, 0, 15, head_width=5, head_length=5, fc='red', ec='red')
        
        # Displaying the scatter plots for each band against dates
        bands = ['B4', 'B3', 'B2']
        colors = ['r', 'g', 'b']
        for band, color in zip(bands, colors):
            axs[1].scatter(df.index, df[band], label=f'{band}', color=color)
        
        # Adding a vertical line that moves across the scatter plots
        current_date = df.index[frame]
        plt.suptitle(f"Patch reef 10 {current_date.strftime('%Y-%m-%d')}", fontsize=16)

        axs[1].axvline(x=current_date, color='k', linestyle='--')
        
        axs[1].legend()
        axs[1].set_ylim(0, 1)  # Assuming the values are normalized (0-1)
        
    ani = animation.FuncAnimation(fig, update, frames=len(images), repeat=False)
    
    # Saving the animation as a GIF
    writer = animation.PillowWriter(fps=fps)  # Adjust fps to your needs
    ani.save(filename, writer=writer)



def plot_image_with_dot_and_arrow(image, x, y, dx=0, dy=-20):
    """
    Plots an image with a red dot and an arrow pointing towards the dot at specified coordinates.
    
    Parameters:
    - image: np.array
        The image to be plotted.
    - x: int
        The x-coordinate where the dot will be placed.
    - y: int
        The y-coordinate where the dot will be placed.
    - dx: int, optional
        The change in x for the arrow direction.
    - dy: int, optional
        The change in y for the arrow direction. Negative values point the arrow upwards.
    """
    
    # Creating a copy of the image to avoid modifying the original
    img_copy = np.copy(image)
    
    # Creating a plot
    fig, ax = plt.subplots()
    
    # Displaying the image
    ax.imshow(img_copy)
    
    # Adding a red dot at the specified coordinates
    ax.plot(x, y, 'ro')
    
    # Adding a red arrow pointing towards the dot
    ax.arrow(x+dx, y - dy *2, -dx, dy, head_width=5, head_length=5, fc='red', ec='red')
    
    # Displaying the plot
    plt.show()

def plot_spectra_with_dates(spectra, dates, title="Spectra across time slices", labels=None,figsize = (10,10), save=False, path=None):
    """
    Plots the spectra for a single point across all time slices.
    
    Parameters:
    - spectra: numpy array of shape (t, z)
    - dates: pandas DatetimeIndex object.
    
    Note:
    Assumes t is time and z represents channels.
    """
    fig, ax = plt.subplots(1, figsize = figsize)
    if len(spectra.shape) != 2:
        raise ValueError("Input spectra should be of shape (t, z)")
    
    if len(dates) != spectra.shape[0]:
        raise ValueError("Length of dates should match the number of time slices in spectra.")
    
    # Converting dates to strings for better display on the x-axis
    date_strings = dates.strftime('%Y-%m-%d')
    
    t, z = spectra.shape
    if labels is not None:
        for channel in range(z):
            plt.plot(date_strings, spectra[:, channel], label=f"Band {labels.get(channel, channel)}")
    
    plt.xticks(rotation=45)
    plt.xlabel('Date')
    plt.ylabel('Intensity')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    
    if save and path:
        plt.savefig(path)
    else:
        plt.show()