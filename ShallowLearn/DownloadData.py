from ShallowLearn.api_download import get_credentials
import pandas as pd
from cdsetool.query import query_features, shape_to_wkt
from cdsetool.credentials import Credentials
from cdsetool.download import download_features
from cdsetool.monitor import StatusMonitor
from datetime import date
from shapely.geometry import box

import requests
import os


top_left_y, top_left_x = -14.4626,145.1453
btm_right_y, btm_right_x = -15.4558,146.1558
bbox = box(top_left_x, top_left_y, btm_right_x, btm_right_y)


tile_ids = ["TLCD55"]

# Define your search parameters
search_terms = {
    "startDate": "2015-01-01",
    "completionDate": "2024-12-31",
    "processingLevel": "S2MSI1C",
    "geometry": bbox,
    "tileId": "55LCD"

}

# Query Sentinel-2 features
features = list(query_features("Sentinel2", search_terms))
save_path = "/mnt/sda_mount/L1C_Full/"
creds = get_credentials()
def filter_features_by_baseline(features):
    # Group features by title to identify duplicates
    grouped_features = {}
    for feature in features:
        title = feature['properties']['title']
        baseline = feature['properties']['processingBaseline']
        if title not in grouped_features:
            grouped_features[title] = feature
        else:
            if baseline > grouped_features[title]['properties']['processingBaseline']:
                grouped_features[title] = feature
    return list(grouped_features.values())
features = filter_features_by_baseline(features)
def filter_features_by_size(features):
    sizes = [feature['properties']['services']['download']['size'] for feature in features if 'size' in feature['properties']['services']['download']]
    average_size = sum(sizes) / len(sizes)
    return [feature for feature in features if feature['properties']['services']['download']['size'] >= average_size * 0.5]
features = filter_features_by_size(features)
total_size = 0
for feature in features:
    services = feature['properties'].get('services', {})
    download_info = services.get('download', {})
    size = download_info.get('size', 0)
    total_size += size

# Convert the size to GB for readability
total_size_gb = total_size / (1024 ** 3)
print(f"Estimated total size of the dataset: {total_size_gb:.2f} GB")


thumbnail_dir = "/mnt/sda_mount/L1C_Full/thumbnails"
os.makedirs(thumbnail_dir, exist_ok=True)

for feature in features:
    thumbnail_url = feature['properties'].get('thumbnail')
    title = feature['properties'].get('title')
    if thumbnail_url and title:
        response = requests.get(thumbnail_url)
        if response.status_code == 200:
            image_path = os.path.join(thumbnail_dir, f"{title}.jpg")
            with open(image_path, 'wb') as file:
                file.write(response.content)
            print(f"Downloaded thumbnail for feature {title}")
        else:
            print(f"Failed to download thumbnail for feature {title}")
    else:
        print(f"No thumbnail URL or title found for feature {feature['id']}")


# Extract relevant feature properties into a list of dictionaries
data = []
for feature in features:
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

# Create a DataFrame from the list of dictionaries
df = pd.DataFrame(data)

# Display the DataFrame
df.head()


list(
    download_features(
        features,
        save_path,
        {
            "concurrency": 4,
            "monitor": StatusMonitor(),
            "credentials": Credentials(creds['sentinel_username'], creds['sentinel_password']),
            
        },
    )
)