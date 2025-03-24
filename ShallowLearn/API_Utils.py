from cdsetool.query import query_features, shape_to_wkt

MIN_SIZE = 300 * 1024 * 1024  # 300MB in bytes
MAX_SIZE = 2 * 1024 * 1024 * 1024  # 2GB in bytes

def filter_by_label(dict_list, labels, target_label):
    """
    Filter a list of dictionaries based on class labels at corresponding indices
    
    Args:
        dict_list (list): List of dictionaries to filter
        labels (list): List of class labels (0,1,2,3 etc)
        target_label (int): The label to filter by
    
    Returns:
        list: Filtered list of dictionaries
    """
    return [dict_list[i] for i, label in enumerate(labels) if label == target_label]


def filter_by_indices(dict_list, indices):
    return [dict_list[i] for i in indices]

def generate_filtered_query(tile, min_size = None, max_size = None):
    
    features = generate_query(tile)
    print(f"{len(features)} tiles found...")
    if min_size is None:
        min_size = MIN_SIZE
    if max_size is None:
        max_size = MAX_SIZE

    features = filter_features_by_size(features, min_size, max_size)
    print(f"{len(features)} tiles found within size range...")
    features = remove_duplicate_features(features)
    print(f"{len(features)} tiles after removing duplicates...")

    return features

def generate_query(tile):
    search_terms_current = {
        "startDate": "2012-01-23",
        "completionDate": "2024-12-31",
        "processingLevel": "S2MSI1C",
        "processingBaseline": "05.11",
        # "geometry": bbox,
        "tileId": tile
    }

    # Intermediate data (Processing Baseline 05.10)
    search_terms_intermediate = {
        "startDate": "2022-01-13",
        "completionDate": "2025-01-27",
        "processingLevel": "S2MSI1C",
        "processingBaseline": "05.10",
        # "geometry": bbox,
        "tileId": tile
    }

    # Historical data 2022-2023 (Processing Baseline 05.09)
    search_terms_historical = {
        "startDate": "2022-04-29",
        "completionDate": "2024-03-13",
        "processingLevel": "S2MSI1C",
        "processingBaseline": "05.09",
        # "geometry": bbox,
        "tileId": tile
    }

    # Earlier historical data (Processing Baseline 05.00)
    search_terms_historical_prior = {
        "startDate": "2015-01-31",
        "completionDate": "2022-04-28",
        "processingLevel": "S2MSI1C",
        "processingBaseline": "05.00",
        # "geometry": bbox,
        "tileId": tile
    }



    # Query all features and combine results
    features_current = list(query_features("Sentinel2", search_terms_current))
    features_intermediate_1 = list(query_features("Sentinel2", search_terms_intermediate))
    features_intermediate_2 = list(query_features("Sentinel2", search_terms_historical))
    features_historical = list(query_features("Sentinel2", search_terms_historical_prior))

    # Combine all features
    features =  features_current + features_intermediate_1 + features_intermediate_2 + features_historical
    return features

def filter_features_by_size(features, min_size=None, max_size=None):
    """
    Filter features based on file size in bytes.
    
    Args:
        features (list): List of feature dictionaries
        min_size (int, optional): Minimum file size in bytes
        max_size (int, optional): Maximum file size in bytes
    
    Returns:
        list: Filtered features
    """
    filtered_features = []
    
    for feature in features:
        try:
            # Get file size from the feature dictionary
            file_size = feature['properties']['services']['download']['size']
            
            # Check if size is within specified range
            size_ok = True
            if min_size is not None and file_size < min_size:
                size_ok = False
            if max_size is not None and file_size > max_size:
                size_ok = False
                
            if size_ok:
                filtered_features.append(feature)
                
        except KeyError:
            # Skip features that don't have size information
            continue
    
    return filtered_features

def remove_duplicate_features(features):
    # Create dictionary to store unique features by date
    unique_dict = {}
    
    for feature in features:
        # Extract start date and processing baseline
        start_date = feature['properties']['startDate']
        baseline = feature['properties'].get('processingBaseline', 0)
        
        # Check if date exists and compare baselines
        if start_date in unique_dict:
            if baseline > unique_dict[start_date][1]:
                unique_dict[start_date] = (feature, baseline)
        else:
            unique_dict[start_date] = (feature, baseline)
    
    # Extract only the features from the dictionary
    cleaned_features = [item[0] for item in unique_dict.values()]
    
    return cleaned_features


def plot_sentinel_features(features):
    import pandas as pd
    import matplotlib.pyplot as plt
    # Convert features to dataframe
    data = []
    for feature in features:
        props = feature['properties']
        data.append({
            'id': feature['id'],
            'startDate': props['startDate'],
            'completionDate': props['completionDate'],
            'cloudCover': props.get('cloudCover', None),
            'processingBaseline': props.get('processingBaseline', None),
            'title': props.get('title', None)
        })
    
    # Create dataframe and convert dates
    df = pd.DataFrame(data)
    df['startDate'] = pd.to_datetime(df['startDate'])
    df['completionDate'] = pd.to_datetime(df['completionDate'])
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(df['startDate'], [1] * len(df), 'o', label='Start Date')
    plt.plot(df['completionDate'], [1] * len(df), 'x', label='Completion Date')
    
    # Customize plot
    plt.xlabel('Time')
    plt.title('Feature Times')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return df
