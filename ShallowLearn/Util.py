import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np

def plot_images_with_info(all_images, band_order = [0,1,2]):
    """
    Plots 10 random images and their coordinates info on subplots from a larger list of images.

    Parameters:
    - all_images: list of lists, where each inner list contains [image_data, x, y, z]
      image_data is an ndarray representing the image, x and y are coordinates, and z is the number of channels.
    """
    # Check if there are at least 10 images
    if len(all_images) < 10:
        raise ValueError("Not enough images available to select 10 random ones.")
    
    # Select 10 random indices from the list
    random_indices = np.random.choice(len(all_images), size=10, replace=False)
    selected_images = [all_images[i] for i in random_indices]

    # Set the number of subplots
    fig, axes = plt.subplots(1, 10, figsize=(20, 4))  # Adjust figure size as needed

    for i, image in enumerate(selected_images):
        ax = axes[i] if len(selected_images) > 1 else axes  # Handle the case of a single subplot

        # Display the image
        if image.shape[-1] == 1:
            ax.imshow(image)
        else:
            ax.imshow(image[:,:,band_order])
        ax.axis('off')  # Turn off axis


    plt.tight_layout()
    plt.show()


def plot_images_on_scatter(transformed_data, imagery, image_fraction=0.11, title = "Images on scatter"):
    """
    Plots images on a scatter plot at specified coordinates.

    Parameters:
    - transformed_data: np.array, an array of coordinates (shape [n, 2]).
    - imagery: list, a list of images in ndarray format.
    - image_fraction: float, the fraction of the plot range to determine the size of the images.
    """
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.scatter(transformed_data[:, 0], transformed_data[:, 1])

    for i in range(len(imagery)):
        # Load and process the image
        if imagery[i].shape[-1] == 1:
            image_data = imagery[i][:,:,:]
        else:
            image_data = imagery[i][:,:,[0,1,2]]
        # Get the corresponding scatter point
        x, y = transformed_data[i, 0], transformed_data[i, 1]

        # Create an OffsetImage and AnnotationBbox
        imagebox = OffsetImage(image_data, zoom=image_fraction)
        ab = AnnotationBbox(imagebox, (x, y), frameon=False)
        
        # Add the AnnotationBbox to the plot
        ax.add_artist(ab)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(title)
    plt.show()


def convert_data_types(df):
    """
    Convert columns in a DataFrame from string to the most appropriate datatype
    (numeric, datetime, or string) by checking if the content fits known date formats,
    then checking for numeric values, and defaulting to string if neither fits.
    Also, consolidates columns with 'FLAG' in their name with corresponding columns
    without 'FLAG'.

    :param df: pandas DataFrame with all columns as strings
    :return: DataFrame with converted datatypes
    """
    # Consolidate columns with similar names where one contains 'FLAG'
    flag_columns = [col for col in df.columns if 'FLAG' in col]
    for flag_col in flag_columns:
        base_col = flag_col.replace('_FLAG', '')
        if base_col in df.columns:
            # Combine data prioritizing non-FLAG data if not NaN
            df[base_col] = df[base_col].combine_first(df[flag_col])
            df.drop(flag_col, axis=1, inplace=True)  # Remove the FLAG column

    # Convert data types
    for column in df.columns:
        # First check for date by trying common date formats
        if pd.to_datetime(df[column], errors='coerce', format='%Y-%m-%d').notna().all():
            df[column] = pd.to_datetime(df[column], format='%Y-%m-%d')
            print(f"Column '{column}' converted to datetime.")
        elif pd.to_datetime(df[column], errors='coerce', format='%m/%d/%Y').notna().all():
            df[column] = pd.to_datetime(df[column], format='%m/%d/%Y')
            print(f"Column '{column}' converted to datetime.")
        else:
            # Attempt to convert to numeric if it's not a recognizable date
            try:
                df[column] = pd.to_numeric(df[column])
                print(f"Column '{column}' converted to numeric.")
            except ValueError:
                print(f"Column '{column}' retained as string.")  # Retain as string if it's neither datetime nor numeric

    return df


def plot_cloud_coverage_over_time(df):
    """
    Plot cloud coverage assessment over time.

    :param df: DataFrame containing the cloud coverage and date of data take.
    """
    # Convert the date column to datetime if not already done
    df['DATATAKE_1_DATATAKE_SENSING_START'] = pd.to_datetime(df['DATATAKE_1_DATATAKE_SENSING_START'])

    # Sort DataFrame by date for better plotting
    df = df.sort_values('DATATAKE_1_DATATAKE_SENSING_START')

    # Creating the plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='DATATAKE_1_DATATAKE_SENSING_START', y='CLOUD_COVERAGE_ASSESSMENT', data=df)
    plt.title('Cloud Coverage Assessment Over Time')
    plt.xlabel('Date of Data Take')
    plt.ylabel('Cloud Coverage (%)')
    plt.xticks(rotation=45)  # Rotate date labels for better visibility
    plt.tight_layout()  # Adjust layout to make room for date labels
    plt.show()

def plot_cloud_coverage_by_processing_baseline(df):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='PROCESSING_BASELINE', y='CLOUD_COVERAGE_ASSESSMENT', data=df)
    plt.title('Cloud Coverage by Processing Baseline')
    plt.xlabel('Processing Baseline')
    plt.ylabel('Cloud Coverage (%)')
    plt.show()

def plot_cloud_coverage_over_time_with_baseline(df):
    """
    Plot cloud coverage over time, color-coded by processing baseline.

    :param df: DataFrame containing the cloud coverage, date of data take, and processing baseline.
    """
    # Ensure 'DATATAKE_1_DATATAKE_SENSING_START' is in datetime format
    df['DATATAKE_1_DATATAKE_SENSING_START'] = pd.to_datetime(df['DATATAKE_1_DATATAKE_SENSING_START'])

    # Ensure 'PROCESSING_BASELINE' is numeric if not already
    df['PROCESSING_BASELINE'] = pd.to_numeric(df['PROCESSING_BASELINE'], errors='coerce')

    # Creating the plot
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='DATATAKE_1_DATATAKE_SENSING_START', y='CLOUD_COVERAGE_ASSESSMENT', 
                 hue='PROCESSING_BASELINE', palette='viridis', data=df)
    
    plt.title('Cloud Coverage Assessment Over Time by Processing Baseline')
    plt.xlabel('Date of Data Take')
    plt.ylabel('Cloud Coverage (%)')
    plt.legend(title='Processing Baseline', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()  # Adjust layout to make room for legend and date labels
    plt.show()

def plot_quality_over_time(df):
    """
    Plot various quality flags over time using a 4-axis subplot.

    :param df: DataFrame containing the quality flags and date of data take.
    """
    # Convert 'PRODUCT_START_TIME' to datetime if it's not already
    df['PRODUCT_START_TIME'] = pd.to_datetime(df['PRODUCT_START_TIME'])

    # Setting up the figure and axes
    fig, axs = plt.subplots(4, 1, figsize=(12, 18), sharex=True)

    # Plotting each quality flag
    quality_columns = ['GENERAL_QUALITY', 'GEOMETRIC_QUALITY', 
                       'RADIOMETRIC_QUALITY', 'SENSOR_QUALITY']
    titles = ['General Quality', 'Geometric Quality', 'Radiometric Quality', 'Sensor Quality']

    for i, quality in enumerate(quality_columns):
        sns.lineplot(x='PRODUCT_START_TIME', y=quality, data=df, ax=axs[i], marker='o')
        axs[i].set_title(titles[i])
        axs[i].set_ylabel('Quality Score')

    # Setting common labels
    plt.xlabel('Date of Data Take')
    plt.xticks(rotation=45)
    plt.tight_layout()  # Adjust layout to make room for label
    plt.show()


def plot_quality_over_time_side_by_side(df):
    """
    Plot various quality flags over time using side-by-side subplots, enhanced for publication quality.

    :param df: DataFrame containing the quality flags and date of data take.
    """
    # Ensure 'PRODUCT_START_TIME' is in datetime format
    df['PRODUCT_START_TIME'] = pd.to_datetime(df['PRODUCT_START_TIME'])

    # Setting the style of the plot
    sns.set(style="whitegrid")  # Using a white grid for a clean layout
    plt.rcParams['font.size'] = 12  # Base font size
    plt.rcParams['axes.labelsize'] = 14  # Axis label size
    plt.rcParams['axes.titlesize'] = 16  # Axis title size
    plt.rcParams['xtick.labelsize'] = 12  # X tick label size
    plt.rcParams['ytick.labelsize'] = 12  # Y tick label size
    plt.rcParams['legend.fontsize'] = 12  # Legend font size

    # Setting up the figure and axes
    fig, axs = plt.subplots(1, 4, figsize=(24, 8), dpi=300)  # 1 row, 4 columns, high resolution for publication

    # Plotting each quality flag
    quality_columns = ['GENERAL_QUALITY', 'GEOMETRIC_QUALITY', 
                       'RADIOMETRIC_QUALITY', 'SENSOR_QUALITY']
    titles = ['General Quality', 'Geometric Quality', 'Radiometric Quality', 'Sensor Quality']
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']  # Color palette for differentiation

    for i, quality in enumerate(quality_columns):
        sns.lineplot(x='PRODUCT_START_TIME', y=quality, data=df, ax=axs[i], marker='o', color=colors[i])
        axs[i].set_title(titles[i])
        axs[i].set_xlabel('Date of Data Take')
        axs[i].set_ylabel('Quality Score')
        axs[i].xaxis.set_major_locator(mdates.YearLocator())  # Set major ticks to every year
        axs[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Only show the year in the tick label
        axs[i].tick_params(axis='x', rotation=45)  # Optionally rotate tick labels for better visibility

    # Enhancing the overall aesthetics
    plt.tight_layout(pad=3.0)  # Adjust layout to make room for label
    plt.show()

def clip_image(image, clip_percent=1):
    """
    Apply a 2% clip on either side of the pixel value distribution for a single image.

    Parameters:
    - image: numpy array representing the image
    - clip_percent: percentage to clip on each side of the pixel value distribution (default is 2%)

    Returns:
    - clipped_image: numpy array with clipped pixel values
    """
    # Calculate the lower and upper clip values
    lower_clip = np.percentile(image, clip_percent)
    upper_clip = np.percentile(image, 100 - clip_percent)
    
    # Clip the image
    clipped_image = np.clip(image, lower_clip, upper_clip)
    
    # Normalize the clipped image to the range [0, 1]
    clipped_image = (clipped_image - lower_clip) / ((upper_clip - lower_clip) + 0.001)
    
    return clipped_image