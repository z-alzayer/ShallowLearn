import os
import requests
import pandas as pd
from tqdm import tqdm
from cdsetool.query import query_features, shape_to_wkt
from cdsetool.credentials import Credentials
from cdsetool.download import download_features
from cdsetool.monitor import StatusMonitor

def query_sentinel2_features(bbox, start_date, end_date, processing_level):
    search_terms = {
        "startDate": start_date,
        "completionDate": end_date,
        "processingLevel": processing_level,
        "geometry": bbox
    }
    return list(query_features("Sentinel2", search_terms))

def estimate_dataset_size(features):
    total_size = sum(
        feature['properties'].get('services', {}).get('download', {}).get('size', 0)
        for feature in features
    )
    return total_size / (1024 ** 3)  # Convert bytes to GB

def create_thumbnail_dir(directory="thumbnails"):
    os.makedirs(directory, exist_ok=True)
    return directory

def download_thumbnail(feature, directory):
    thumbnail_url = feature['properties'].get('thumbnail')
    title = feature['properties'].get('title')
    if thumbnail_url and title:
        response = requests.get(thumbnail_url)
        if response.status_code == 200:
            image_path = os.path.join(directory, f"{title}.jpg")
            with open(image_path, 'wb') as file:
                file.write(response.content)
        else:
            print(f"Failed to download thumbnail for feature {title}")
    else:
        print(f"No thumbnail URL or title found for feature {feature['id']}")

def download_thumbnails(features, directory):
    for feature in tqdm(features, desc="Downloading thumbnails"):
        download_thumbnail(feature, directory)

def prepare_features_dataframe(features):
    data = [
        {
            'id': feature['id'],
            'title': feature['properties'].get('title'),
            'startDate': feature['properties'].get('startDate'),
            'completionDate': feature['properties'].get('completionDate'),
            'productType': feature['properties'].get('productType'),
            'processingLevel': feature['properties'].get('processingLevel'),
            'platform': feature['properties'].get('platform'),
            'instrument': feature['properties'].get('instrument'),
            'cloudCover': feature['properties'].get('cloudCover'),
            'geometry': feature['geometry'],
            'thumbnail_url': feature['properties'].get('thumbnail'),
            'download_url': feature['properties'].get('services', {}).get('download', {}).get('url'),
            'size': feature['properties'].get('services', {}).get('download', {}).get('size', 0)
        }
        for feature in features
    ]
    return pd.DataFrame(data)

def save_dataframe_to_csv(df, filename="sentinel2_features.csv"):
    df.to_csv(filename, index=False)

def download_full_features(features, file_path, creds):
    credentials = Credentials(creds['sentinel_username'], creds['sentinel_password'])
    for feature in tqdm(features, desc="Downloading full features"):
        list(
            download_features(
                [feature],
                file_path,
                {
                    "concurrency": 4,
                    "monitor": StatusMonitor(),
                    "credentials": credentials,
                },
            )
        )

def main():
    bbox = "your bounding box or shapefile converted to WKT"
    start_date = "2015-01-01"
    end_date = "2024-12-31"
    processing_level = "S2MSI2A"
    
    print("Querying Sentinel-2 features...")
    features = query_sentinel2_features(bbox, start_date, end_date, processing_level)
    
    print("Estimating dataset size...")
    total_size_gb = estimate_dataset_size(features)
    print(f"Estimated total size of the dataset: {total_size_gb:.2f} GB")
    
    print("Creating thumbnail directory...")
    thumbnail_dir = create_thumbnail_dir()
    
    print("Downloading thumbnails...")
    download_thumbnails(features, thumbnail_dir)
    
    print("Preparing DataFrame...")
    df = prepare_features_dataframe(features)
    
    print("Saving DataFrame to CSV...")
    save_dataframe_to_csv(df)
    
    creds = {'sentinel_username': 'your_username', 'sentinel_password': 'your_password'}
    file_path = "/path/to/save/full_features"
    
    print("Downloading full features...")
    download_full_features(features, file_path, creds)
    
    print("Process complete.")

if __name__ == "__main__":
    main()
