import zipfile
import os
import re
from collections import defaultdict
from collections import OrderedDict
import shutil
from ShallowLearn.band_mapping import band_mapping



def filter_files(file_paths, min_size_mb=1, max_size_mb=10):
    """
    Filters out files smaller than min_size_mb or larger than max_size_mb.
    
    :param file_paths: List of file paths to check.
    :param min_size_mb: Minimum allowed file size in MB.
    :param max_size_mb: Maximum allowed file size in MB.
    :return: List of file paths that satisfy the size conditions.
    """
    filtered_files = []
    
    for file_path in file_paths:
        # Get file size in bytes
        file_size_bytes = os.path.getsize(file_path)
        
        # Convert file size to MB
        file_size_mb = file_size_bytes / (1024 * 1024)
        
        # Print file sizes
        # print(f"File: {file_path}, Size: {file_size_mb:.2f} MB")
        
        # Keep the file if it satisfies the size conditions
        if min_size_mb <= file_size_mb <= max_size_mb:
            filtered_files.append(file_path)
            
    return filtered_files



def get_files_with_extension(directory_path, extension):
    return [file for file in os.listdir(directory_path) if file.endswith(extension)]


def unzip_files(zip_file_path, file_names, extract_dir):
    """
    Unzips specific files from a zip file.

    zip_file_path: Path to the zip file.
    file_names: List of specific files to extract.
    extract_dir: Directory where files will be extracted.
    """
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        for file in file_names:
            if file in zip_ref.namelist():
                zip_ref.extract(file, extract_dir)
            else:
                print(f"File {file} not found in the zip file.")

def get_file_names_from_zip(zip_file_path):
    """
    Returns a list of file names from a zip file.

    zip_file_path: Path to the zip file.
    """
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        return zip_ref.namelist()
    
def delete_files_from_dir(directory, files_to_keep=None):
    """
    Deletes all files and subdirectories from a directory, except for the specified files to keep.

    directory: Directory from which files and subdirectories will be deleted.
    files_to_keep: List of filenames to keep (optional).
    """
    if files_to_keep is None:
        files_to_keep = []

    for root, dirs, files in os.walk(directory, topdown=False):
        for filename in files:
            file_path = os.path.join(root, filename)
            if filename not in files_to_keep:
                os.remove(file_path)
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            shutil.rmtree(dir_path)

def list_files_in_dir_recur(directory):
    """
    Lists all files in a directory including subdirectories.

    directory: Directory from which files will be listed.
    """
    files = []

    # Iterate over items in the directory
    for root, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            files.append(os.path.join(root, filename))

    return files



def list_files_in_dir(directory):
    """
    Lists all files in a directory.

    directory: Directory from which files will be listed.
    """
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    return files

def filter_files_by_extension(file_list, extension):
    """
    Filters a list of files for a specific file extension.

    file_list: List of file names.
    extension: File extension to filter by.
    """
    return [f for f in file_list if os.path.splitext(f)[1] == extension]

def check_in_string(input_string, extension=None, other_string=None):
    """
    Checks if a string ends with a specific extension and contains a specific substring.

    Parameters:
    - input_string (str): The string to check.
    - extension (str): The extension to check for.
    - other_string (str): The substring to check for.
    
    """
    extension_present = False
    other_string_present = False

    if extension is not None:
        extension_present = input_string.endswith(extension)

    if other_string is not None:
        other_string_present = other_string in input_string

    return extension_present and (other_string_present if other_string is not None else True)



def check_values_in_filenames(file_list, values):
    """
    Checks if any of the provided values are found in the names of the files.

    file_list: List of file names.
    values: List of values to check in file names.
    """
    return [file_name for file_name in file_list if any(value in file_name for value in values)]



def get_highest_resolutions(files):
    """
    Get the highest resolution for each band from a list of Sentinel-2 file names.

    Parameters:
    - files (list): A list of Sentinel-2 file names.

    Returns:
    - highest_resolutions (defaultdict): A defaultdict containing the highest resolution
      for each band found in the file names.
    """
    highest_resolutions = defaultdict(lambda: (float("inf"), ""))

    # Compile regex for performance
    resolution_regex = re.compile(r"R(\d+)m")
    band_regex = re.compile(r"(B[\dA]+)_(\d+)m")

    for file in files:
        # Extract resolution from directory
        resolution_match = resolution_regex.search(file)
        if resolution_match is not None:
            resolution = int(resolution_match.group(1))

        # Extract band and resolution from filename
        band_match = band_regex.search(file)
        if band_match is not None:
            band = band_match.group(1)
            band_resolution = int(band_match.group(2))

            # Check for inconsistencies between directory and filename
            if resolution != band_resolution:
                print(f"Warning: Inconsistent resolutions in {file}")

            # If this is the highest resolution found for this band, store it
            if resolution < highest_resolutions[band][0]:
                highest_resolutions[band] = (resolution, file)

    # Only return the file paths
    return [file for resolution, file in highest_resolutions.values()]

def order_band_names_noreg(file_names, band_mapping = band_mapping):
    # Extracts band identifier from file name
    def get_band_identifier(file_name):
        for band in band_mapping:
            if band in file_name:
                return band
        return None

    # Sort by the index value in band_mapping
    return sorted(file_names, key=lambda x: band_mapping[get_band_identifier(x)]['index'])


def order_by_band(files, order = band_mapping.keys()):
    """
    Function to order image files by band. The default order is provided by the band_mapping keys.
    
    Parameters
    ----------
    files : list
        A list of file paths to the image files.
    order : iterable, optional
        An iterable specifying the desired order of bands. Defaults to the keys of the band_mapping dictionary.

    Returns
    -------
    list
        A list of file paths ordered by band according to the specified order.

    Notes
    -----
    This function specifically looks for files with highest resolution and orders them.
    Band information is extracted from the filenames assuming a specific pattern: "B[number][optional letter]_[resolution]m".
    """
    # Get highest resolution files
    highest_resolution_files = get_highest_resolutions(files)
    
    # Extract bands from filenames
    bands_files = {re.search(r"(B[\dA]+)_(\d+)m", file).group(1): file for file in highest_resolution_files}
    
    # Create a new OrderedDict
    ordered_files = OrderedDict()
    
    # Populate the OrderedDict
    for band in order:
        if band in bands_files:
            ordered_files[band] = bands_files[band]
            
    # Return only the file paths
    return list(ordered_files.values())


def extract_pvi_images(directory):
    """ Takes a directory and returns all PVI images within directory with their paths"""
    PVI_Files = list_files_in_dir_recur(directory)
    PVI = [i for i in PVI_Files if "PVI" in i and i.endswith(".jp2")]

    return PVI 

def extract_MTD_files(directory):
    files = list_files_in_dir_recur(directory)
    files = [file for file in files if file.endswith(".xml") and "MTD_MSIL1C" in file]
    return files