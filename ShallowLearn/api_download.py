from landsatxplore.api import API
from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
import abc
from datetime import date
import os
from shapely.geometry import box


class Creds():
    landsat_username = os.environ.get('LSAT_USER')
    landsat_password = os.environ.get('LSAT_PASS')
    sentinel_username = os.environ.get('SEN_USER')
    sentinel_password = os.environ.get('SEN_PASS')

class DownloadFiles(metaclass=abc.ABCMeta):
    """Abstract base class for downloading satellite imagery files."""

    @abc.abstractmethod
    def __init__(self, username, password):
        """Initialize the downloader with the given username and password."""
        pass

    @abc.abstractmethod
    def download(self, dates, cloud_cover):
        """Download satellite imagery files based on the given dates and cloud cover."""
        pass


class LandSatDownload(DownloadFiles):
    """Concrete class for downloading Landsat satellite imagery files."""

    def __init__(self):
        """Initialize the Landsat downloader with the given username and password."""
        self.username = os.environ.get('LSAT_USER')
        self.password = os.environ.get('LSAT_PASS')

    def download(self, dataset:str, location:list[float], dates:list[str], cloud_cover:int):
        """Download Landsat products using lat_long and a date range.
        
        Args:
            dataset (str): The name of the dataset to search.
            location (list[float]): The latitude and longitude of the desired location.
            dates (list[str]): The start and end dates for the search.
            cloud_cover (int): The maximum cloud cover percentage allowed.
        
        Returns:
            list: A list of scenes that match the search criteria.
        """
        api = API(self.username, self.password)

        scenes = api.search(
            dataset=dataset,
            latitude=location[0],
            longitude=location[1],
            start_date=dates[0],
            end_date=dates[1],
            max_cloud_cover=cloud_cover
        )

        api.logout()
        return scenes

    def download_from_json(self, dataset:str, location, dates:list[str], cloud_cover:int):
        """Download Landsat products using a GeoJSON file and a date range.
        
        Args:
            dataset (str): The name of the dataset to search.
            location (str): The path to the GeoJSON file.
            dates (list[str]): The start and end dates for the search.
            cloud_cover (int): The maximum cloud cover percentage allowed.
        
        Returns:
            list: A list of scenes that match the search criteria.
        """
        api = API(self.username, self.password)

        footprint = read_geojson(location)
        # Access the coordinates
        coordinates = footprint["features"][0]["geometry"]["coordinates"][0]

        # Extract the individual coordinates
        min_longitude, min_latitude = [145.141999, -14.465811]
        max_longitude, max_latitude = [146.159657, -15.460986]
        # print(footprint)
        scenes = api.search(
            dataset=dataset,
            start_date=dates[0],
            end_date=dates[1],
            max_cloud_cover=cloud_cover,
            bbox = (min_longitude, min_latitude, max_longitude, max_latitude)
        )
        
        api.logout()
        return scenes

class SentinelSatDownload(DownloadFiles):
    """Concrete class for downloading Sentinel-2 satellite imagery files."""

    def __init__(self, username, password):
        """Initialize the Sentinel-2 downloader with the given username and password."""
        self.username = username
        self.password = password
    
    def download(self, bbox, dates:list[str], cloud_cover:[int], path:str):
        """Download Sentinel-2 products using a bounding box and a date range.
        
        Args:
            bbox: The bounding box (in WKT or GeoJSON format) of the area of interest.
            dates (list[str]): The start and end dates for the search.
            cloud_cover (list[int]): The minimum and maximum cloud cover percentage allowed.
        """
        api = SentinelAPI(self.username, self.password, 'https://scihub.copernicus.eu/dhus')
        print(api)
        # search for products
        products = api.query(bbox,
                            date=(dates[0], dates[1]),
                            platformname='Sentinel-2',
                            producttype='S2MSI2A',
                            cloudcoverpercentage=(cloud_cover[0], cloud_cover[1]))

        # download the products

        api.download_all(products, directory_path=path)



def main():
    # Replace with your valid username and password for each API
    landsat_username = os.environ.get('LSAT_USER')
    landsat_password = os.environ.get('LSAT_PASS')
    sentinel_username = os.environ.get('SEN_USER')
    sentinel_password = os.environ.get('SEN_PASS')
    path = "/home/zba21/Documents/ShallowLearn/Data/Tile_TLCD55.geojson"
    # Latitude and longitude for a location near London
    london_lat, london_lon = 16.0959,-86.9362

    # Date range for the search YMD format for sentinel-2 - fix for landsat eventually probably
    start_date = '20150101'
    end_date = '20220531'

    # Maximum cloud cover percentage for Landsat
    max_cloud_cover_landsat = 60

    # Cloud cover range for Sentinel-2
    min_cloud_cover_sentinel = 0
    max_cloud_cover_sentinel = 25

    # # Instantiate the Landsat downloader and search for scenes
    landsat_downloader = LandSatDownload()
    landsat_scenes = landsat_downloader.download_from_json('landsat_ot_c2_l1', path,[start_date, end_date], 10)
    # print("Landsat scenes:", landsat_scenes)
    print(len(landsat_scenes))
    output = "/mnt/sda_mount/Landsat/"
    from landsatxplore.earthexplorer import EarthExplorer
    counter = 0
    for scene in landsat_scenes:
        print(scene["landsat_product_id"])
        if scene['nadir-off_nadir'] == "NADIR":
            ee = EarthExplorer(landsat_username, landsat_password)

            ee.download(scene["landsat_scene_id"], output_dir=output)
            counter +=1
            print(counter)
        print("Scene download completed - next one")
    ee.logout()

    # # Instantiate the Sentinel-2 downloader and search for products
    # sentinel_downloader = SentinelSatDownload(sentinel_username, sentinel_password)

    # Create a bounding box around the location (approx. 1 degree in each direction)
    # bbox = f"POLYGON(({london_lon - 0.001} {london_lat - 0.001}, {london_lon - 0.001} {london_lat + 0.001}, {london_lon + 0.001} {london_lat + 0.001}, {london_lon + 0.001} {london_lat - 0.001}, {london_lon - 0.001} {london_lat - 0.001}))"
    
    # sentinel_products = sentinel_downloader.download(bbox, [start_date, end_date], [min_cloud_cover_sentinel, max_cloud_cover_sentinel], path = "/media/ziad/Expansion/Honduras/")
    # print("Sentinel-2 products:", sentinel_products)

if __name__ == '__main__':
    main()
    
