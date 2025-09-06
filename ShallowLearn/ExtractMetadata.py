import pandas as pd
import re
import requests
import os

# LoadData module deprecated - functionality moved to ShallowLearn.io
# import ShallowLearn.LoadData as ld
import ShallowLearn.FileProcessing as fp
import ShallowLearn.Util as utilities
import ShallowLearn.QuickLook as quicklook


def map_and_filter_columns(df):
    # Dictionary mapping between the two column sets
    df.columns = [i.upper() for i in df.columns]
    column_mapping = {
        'ID': 'DATATAKE_1_ID',
        'TITLE':"COMMON_COMPONENT",
        'STARTDATE': 'DATATAKE_1_DATATAKE_SENSING_START',
        'COMPLETIONDATE': 'PRODUCT_STOP_TIME',
        'PRODUCTTYPE': 'PRODUCT_TYPE',
        'PROCESSINGLEVEL': 'PROCESSING_LEVEL',
        'PLATFORM': 'DATATAKE_1_SPACECRAFT_NAME',
        'CLOUDCOVER': 'CLOUD_COVERAGE_ASSESSMENT',
        'THUMBNAIL_URL': 'PREVIEW_IMAGE_URL'
    }
    
    # Create a new dataframe with only the mapped columns
    mapped_columns = {old_col: new_col for old_col, new_col in column_mapping.items() 
                     if old_col in df.columns}
    
    if not mapped_columns:
        return None
    
    # Create new dataframe with renamed columns
    result_df = df[list(mapped_columns.keys())].copy()
    result_df = result_df.rename(columns=mapped_columns)
    
    return result_df

def download_thumbnails_helper(thumbnail_url, title, thumbnail_dir):
        response = requests.get(thumbnail_url)
        if response.status_code == 200:
            image_path = os.path.join(thumbnail_dir, f"{title}.jpg")
            if os.path.exists(image_path):
                return f"{title}.jpg already downloaded"
            with open(image_path, 'wb') as file:
                file.write(response.content)
            print(f"Downloaded thumbnail for feature {title}")
        else:
            print(f"Failed to download thumbnail for feature {title}")
        return f"{title}.jpg"

def generate_api_df(features, thumbnail_dir, download_thumbnails = True):
    data = []
    for feature in features:
        thumbnail_url = feature['properties'].get('thumbnail')
        title = feature['properties'].get('title')
        properties = feature['properties']
        data.append({
            'id': feature['id'],
            'title': properties.get('title'),
            'startDate': properties.get('startDate'),
            'completionDate': properties.get('completionDate'),
            'productType': properties.get('productType'),
            'processingLevel': properties.get('processingLevel'),
            'platform': properties.get('platform'),
            'instrument': properties.get('instrument'),
            'cloudCover': properties.get('cloudCover'),
            'geometry': feature['geometry'],
            'thumbnail_url': properties.get('thumbnail'),
            'download_url': properties.get('services', {}).get('download', {}).get('url'),
            'size': properties.get('services', {}).get('download', {}).get('size', 0)
        })
        if (thumbnail_url and title) and download_thumbnails:
            download_thumbnails_helper(thumbnail_url, title, thumbnail_dir)
    df = pd.DataFrame(data)

    
    return df



def generate_metadata_dataframe(directory, gen_from_zips = False):
    """Generates a dataframe from the metadata of all of the imagery"""
    print(directory)
    if gen_from_zips is False:
        mtd_file_paths = fp.extract_MTD_files(directory)
    else:
        mtd_file_paths = directory
    metadata = {}
    for file in mtd_file_paths:
        data_loader = ld.LoadSentinel2L1C(file)
        subdata = data_loader.load()
        metadata[file] = data_loader.tags
    df = pd.DataFrame(metadata).T
    df.reset_index(inplace = True)
    df.rename(columns={df.columns[0]: 'FILE_PATH'}, inplace=True)
    df = utilities.convert_data_types(df)
    
    return df

def combine_metadata_w_pvi_analysis(directory, quick_look, verbose=False, gen_from_zips = False):
    df = generate_metadata_dataframe(directory, gen_from_zips = gen_from_zips)
    print(df)
    # Extract common component for dataframe and file_list
    df['COMMON_COMPONENT'] = df['FILE_PATH'].apply(extract_common_component)
    common_components_list = [extract_common_component(path) for path in quick_look.files]

    # Create a dictionary for mapping the order based on file_list
    order_mapping = {component: i for i, component in enumerate(common_components_list)}

    # Ensure that all components in the dataframe match the components in the file list
    missing_components = df[~df['COMMON_COMPONENT'].isin(common_components_list)]
    if not missing_components.empty:
        if verbose:
            print(f"Missing components in the file list: {missing_components['COMMON_COMPONENT'].tolist()}")
        df = df[df['COMMON_COMPONENT'].isin(common_components_list)]
    
    unmatched_components = [comp for comp in common_components_list if comp not in df['COMMON_COMPONENT'].values]
    if unmatched_components:
        if verbose:
            print(f"Unmatched components in the dataframe: {unmatched_components}")
        # Filter out unmatched components from quick_look attributes
        filtered_files = []
        filtered_labels = []
        filtered_imagery = []
        for file, label, image in zip(quick_look.files, quick_look.labels, quick_look.imagery):
            common_component = extract_common_component(file)
            if common_component not in unmatched_components:
                filtered_files.append(file)
                filtered_labels.append(label)
                filtered_imagery.append(image)
        quick_look.files = filtered_files
        quick_look.labels = filtered_labels
        quick_look.imagery = filtered_imagery
        common_components_list = [extract_common_component(path) for path in quick_look.files]
        order_mapping = {component: i for i, component in enumerate(common_components_list)}
    print(df)
    # Sort the dataframe based on the common component order
    df['ORDER'] = df['COMMON_COMPONENT'].map(order_mapping)
    df_sorted = df.sort_values('ORDER').drop(columns=['ORDER'])
    
    # Verify that the ordering is correct
    for idx, component in enumerate(df_sorted['COMMON_COMPONENT']):
        if component != common_components_list[idx]:
            raise ValueError(f"Mismatch at index {idx}: {component} != {common_components_list[idx]}")

    # Add labels from quick_look
    df_sorted['Label'] = quick_look.labels

    # Additional check to ensure the imagery matches the dataframe
    if len(df_sorted) != len(quick_look.imagery):
        raise ValueError("Mismatch in the number of rows between the sorted dataframe and the quick_look imagery")

    return df_sorted

def extract_common_component(path):
    match = re.search(r'(S2[^/]+\.SAFE)', path)
    return match.group(1) if match else path



def generate_metadata_plots(df):
    utilities.plot_cloud_coverage_over_time(df)
    utilities.plot_cloud_coverage_over_time_with_baseline(df)
    utilities.plot_quality_over_time_side_by_side(df)