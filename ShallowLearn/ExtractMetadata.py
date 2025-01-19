import pandas as pd
import re

import ShallowLearn.LoadData as ld
import ShallowLearn.FileProcessing as fp
import ShallowLearn.Util as utilities
import ShallowLearn.QuickLook as quicklook


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