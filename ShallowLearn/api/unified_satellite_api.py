"""
Unified Satellite API - Basic implementation for testing against existing working files.
This focuses on getting consistent API returns from both Landsat and Sentinel-2.
"""

import os
import json
import requests
from datetime import datetime
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass
from pathlib import Path
from shapely.geometry import box, Point

# Load environment variables directly
from dotenv import load_dotenv
load_dotenv()

try:
    from cdsetool.query import query_features
except ImportError:
    print("Warning: cdsetool not available. Sentinel-2 API will not work.")
    query_features = None


@dataclass
class SatelliteProduct:
    """Standardized satellite product representation"""
    product_id: str
    satellite: str  # 'landsat' or 'sentinel2'
    sensor: str
    acquisition_date: str
    cloud_cover: float
    processing_level: str
    thumbnail_url: Optional[str] = None
    download_url: Optional[str] = None
    bounds: Optional[Dict] = None
    metadata: Optional[Dict] = None

    def to_dict(self) -> Dict:
        return {
            'product_id': self.product_id,
            'satellite': self.satellite,
            'sensor': self.sensor,
            'acquisition_date': self.acquisition_date,
            'cloud_cover': self.cloud_cover,
            'processing_level': self.processing_level,
            'thumbnail_url': self.thumbnail_url,
            'download_url': self.download_url,
            'bounds': self.bounds
        }


@dataclass
class SatelliteQuery:
    """Unified query parameters for satellite data"""
    geometry: Union[Point, Tuple[float, float, float, float]]  # Point(lon, lat) or (lon_min, lat_min, lon_max, lat_max)
    date_range: Tuple[str, str]
    cloud_cover_max: int = 100
    processing_level: str = "L1C"  # L1C|L2A for S2, L1|L2 for Landsat
    satellites: List[str] = None
    max_results: int = 1000

    def __post_init__(self):
        if self.satellites is None:
            self.satellites = ["landsat", "sentinel2"]

    def to_dict(self) -> Dict:
        return {
            'geometry': str(self.geometry),
            'date_range': self.date_range,
            'cloud_cover_max': self.cloud_cover_max,
            'processing_level': self.processing_level,
            'satellites': self.satellites,
            'max_results': self.max_results
        }


class LandsatUSGSDownloader:
    """Landsat downloader using USGS M2M API - matches existing BarAlHikman implementation"""
    
    def __init__(self):
        self.service_url = "https://m2m.cr.usgs.gov/api/api/json/stable/"
        self.datasets = {
            'L1': 'landsat_ot_c2_l1',    # Landsat 8-9 OLI/TIRS Collection 2 Level 1
            'L2': 'landsat_ot_c2_l2',    # Landsat 8-9 OLI/TIRS Collection 2 Level 2
        }
        # Load credentials directly from environment
        self.token = os.getenv('LSAT_TOKEN')
        self.username = os.getenv('LSAT_USER')
        if self.username:
            self.username = self.username.lower()  # Force lowercase for USGS API compatibility
        
        if not self.token or not self.username:
            raise ValueError("Landsat credentials not found. Check .env file.")
    
    def _send_request(self, endpoint: str, data: Dict, api_key: str = None) -> Dict:
        """Send request to USGS M2M API - matches existing implementation"""
        headers = {'Content-Type': 'application/json'}
        if api_key:
            headers['X-Auth-Token'] = api_key
        
        response = requests.post(self.service_url + endpoint, json=data, headers=headers)
        response.raise_for_status()
        output = response.json()
        
        if output.get('errorCode'):
            raise Exception(f"{output['errorCode']}: {output['errorMessage']}")
        return output['data']
    
    def _convert_geometry_to_mbr(self, geometry) -> Dict:
        """Convert geometry to USGS MBR format"""
        if isinstance(geometry, Point):
            # For point queries, create small bounding box around the point
            lon, lat = geometry.x, geometry.y
            buffer = 0.01  # ~1km buffer
            return {
                'filterType': 'mbr',
                'lowerLeft': {'latitude': lat - buffer, 'longitude': lon - buffer},
                'upperRight': {'latitude': lat + buffer, 'longitude': lon + buffer}
            }
        elif isinstance(geometry, tuple) and len(geometry) == 4:
            # Bounding box: (lon_min, lat_min, lon_max, lat_max)
            lon_min, lat_min, lon_max, lat_max = geometry
            return {
                'filterType': 'mbr',
                'lowerLeft': {'latitude': lat_min, 'longitude': lon_min},
                'upperRight': {'latitude': lat_max, 'longitude': lon_max}
            }
        else:
            raise ValueError(f"Unsupported geometry type: {type(geometry)}")
    
    def search(self, query: SatelliteQuery) -> List[SatelliteProduct]:
        """Search Landsat scenes - matches existing BarAlHikman data_download.py logic"""
        print(f"Searching Landsat scenes...")
        
        # Step 1: Authenticate
        auth_payload = {'username': self.username, 'token': self.token}
        api_key = self._send_request("login-token", auth_payload)
        
        try:
            # Step 2: Build search payload using same structure as existing code
            spatial_filter = self._convert_geometry_to_mbr(query.geometry)
            acquisition_filter = {'start': query.date_range[0], 'end': query.date_range[1]}
            cloud_cover_filter = {'min': 0, 'max': query.cloud_cover_max}
            
            dataset_name = self.datasets.get(query.processing_level, self.datasets['L1'])
            
            search_payload = {
                'datasetName': dataset_name,
                'maxResults': query.max_results,
                'startingNumber': 1,
                'sceneFilter': {
                    'spatialFilter': spatial_filter,
                    'acquisitionFilter': acquisition_filter,
                    'cloudCoverFilter': cloud_cover_filter
                }
            }
            
            # Step 3: Execute search
            scenes = self._send_request("scene-search", search_payload, api_key)
            
            if scenes['recordsReturned'] == 0:
                print("No Landsat scenes found for query")
                return []
            
            print(f"Found {scenes['recordsReturned']} Landsat scenes")
            
            # Step 4: Convert to standardized format
            products = []
            for scene in scenes['results']:
                # Extract sensor type from entity ID
                entity_id = scene.get('entityId', '')
                if entity_id.startswith('LC08') or entity_id.startswith('LC09'):
                    sensor = 'OLI/TIRS'
                elif entity_id.startswith('LE07'):
                    sensor = 'ETM+'
                elif entity_id.startswith('LT05') or entity_id.startswith('LT04'):
                    sensor = 'TM'
                else:
                    sensor = 'Unknown'
                
                # Look for browse/thumbnail URL in various possible locations
                thumbnail_url = None
                if 'browse' in scene:
                    browse_info = scene['browse']
                    if isinstance(browse_info, dict):
                        thumbnail_url = browse_info.get('browsePath') or browse_info.get('browseUrl')
                    elif isinstance(browse_info, list) and len(browse_info) > 0:
                        thumbnail_url = browse_info[0].get('browsePath') or browse_info[0].get('browseUrl')
                
                product = SatelliteProduct(
                    product_id=scene.get('entityId'),
                    satellite='landsat',
                    sensor=sensor,
                    acquisition_date=scene.get('temporalCoverage', {}).get('startDate'),
                    cloud_cover=float(scene.get('cloudCover', 0)),
                    processing_level=query.processing_level,
                    thumbnail_url=thumbnail_url,
                    bounds=scene.get('spatialBounds'),
                    metadata=scene
                )
                products.append(product)
            
            return products
            
        finally:
            # Step 5: Logout
            self._send_request("logout", None, api_key)
    
    def download_product(self, product: SatelliteProduct, output_dir: str) -> str:
        """Download a Landsat product using USGS M2M API - matches BarAlHikman implementation"""
        import os
        import re
        import time
        import datetime
        from pathlib import Path
        
        # Step 1: Authenticate
        payload = {'username': self.username, 'token': self.token}
        api_key = self._send_request("login-token", payload)
        
        try:
            # Step 2: Get download options for the scene
            payload = {
                'datasetName': self.datasets.get(product.processing_level, 'landsat_ot_c2_l1'),
                'entityIds': [product.product_id]
            }
            
            download_options = self._send_request("download-options", payload, api_key)
            
            if not download_options:
                raise Exception(f"No download options available for {product.product_id}")
            
            # Find available download option - matches BarAlHikman logic
            download_option = None
            for option in download_options:
                if option.get('available'):
                    download_option = option
                    break
                        
            if not download_option:
                raise Exception(f"No available download options for {product.product_id}")
            
            # Step 3: Request download with label (BarAlHikman pattern)
            label = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            payload = {
                'downloads': [{'entityId': product.product_id, 'productId': download_option['id']}],
                'label': label
            }
            
            request_results = self._send_request("download-request", payload, api_key)
            
            # Step 4: Handle download availability (matches BarAlHikman)
            download_url = None
            
            # If downloads are preparing, poll until ready
            if request_results.get('preparingDownloads'):
                print(f"Download preparing for {product.product_id}...")
                payload = {'label': label}
                max_attempts = 10  # Limit polling attempts
                attempt = 0
                
                while attempt < max_attempts:
                    more_downloads = self._send_request("download-retrieve", payload, api_key)
                    available = more_downloads.get('available', [])
                    
                    if available:
                        download_url = available[0]['url']
                        break
                        
                    print("Waiting for downloads to become available...")
                    time.sleep(30)
                    attempt += 1
                
                if not download_url:
                    raise Exception(f"Download did not become available for {product.product_id}")
            else:
                # Download immediately available
                available_downloads = request_results.get('availableDownloads', [])
                if available_downloads:
                    download_url = available_downloads[0]['url']
                else:
                    raise Exception(f"No download URL available for {product.product_id}")
            
            # Step 5: Download the file using requests (BarAlHikman pattern)
            response = requests.get(download_url, stream=True)
            response.raise_for_status()
            
            # Determine filename from response headers or URL
            disposition = response.headers.get('content-disposition', '')
            filename = None
            if 'filename=' in disposition:
                filename = re.findall("filename=(.+)", disposition)[0].strip('"')
            else:
                filename = download_url.split("/")[-1].split("?")[0]
            
            if not filename:
                filename = f"{product.product_id}.tar.gz"  # Default extension
            
            file_path = os.path.join(output_dir, filename)
            
            # Download with progress indication
            print(f"Downloading {filename}...")
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            print(f"Downloaded {filename}")
            return file_path
            
        finally:
            # Step 5: Logout
            self._send_request("logout", None, api_key)


class Sentinel2CDSEDownloader:
    """Sentinel-2 downloader using CDSE API - matches existing ShallowLearn implementation"""
    
    def __init__(self):
        # Load credentials directly from environment
        self.username = os.getenv('SEN_USER')
        self.password = os.getenv('SEN_PASS')
        
        if not self.username or not self.password:
            raise ValueError("Sentinel-2 credentials not found. Check .env file.")
        
        if query_features is None:
            raise ImportError("cdsetool not available. Cannot use Sentinel-2 API.")
    
    def _convert_geometry_to_bbox(self, geometry):
        """Convert geometry to shapely box for CDSE API"""
        if isinstance(geometry, Point):
            # For point queries, create small bounding box
            lon, lat = geometry.x, geometry.y
            buffer = 0.01
            return box(lon - buffer, lat - buffer, lon + buffer, lat + buffer)
        elif isinstance(geometry, tuple) and len(geometry) == 4:
            # Bounding box: (lon_min, lat_min, lon_max, lat_max)
            lon_min, lat_min, lon_max, lat_max = geometry
            return box(lon_min, lat_min, lon_max, lat_max)
        else:
            raise ValueError(f"Unsupported geometry type: {type(geometry)}")
    
    def search(self, query: SatelliteQuery) -> List[SatelliteProduct]:
        """Search Sentinel-2 scenes - matches existing ShallowLearn DownloadData.py logic"""
        print(f"Searching Sentinel-2 scenes...")
        
        # Convert query parameters to CDSE format
        bbox = self._convert_geometry_to_bbox(query.geometry)
        
        # Map processing levels
        processing_level_map = {
            'L1C': 'S2MSI1C',
            'L2A': 'S2MSI2A'
        }
        processing_level = processing_level_map.get(query.processing_level, 'S2MSI1C')
        
        # Build search terms using same structure as existing code
        search_terms = {
            "startDate": query.date_range[0],
            "completionDate": query.date_range[1],
            "processingLevel": processing_level,
            "geometry": bbox,
            "maxRecords": query.max_results
        }
        
        # Execute search using cdsetool
        features = list(query_features("Sentinel2", search_terms))
        
        if not features:
            print("No Sentinel-2 scenes found for query")
            return []
        
        print(f"Found {len(features)} Sentinel-2 scenes")
        
        # Convert to standardized format
        products = []
        for feature in features:
            props = feature['properties']
            
            product = SatelliteProduct(
                product_id=props.get('title'),
                satellite='sentinel2',
                sensor='MSI',
                acquisition_date=props.get('startDate'),
                cloud_cover=float(props.get('cloudCover', 0)),
                processing_level=query.processing_level,
                thumbnail_url=props.get('thumbnail'),
                download_url=props.get('services', {}).get('download', {}).get('url'),
                bounds=feature.get('geometry'),
                metadata=feature
            )
            products.append(product)
        
        return products
    
    def download_product(self, product: SatelliteProduct, output_dir: str) -> str:
        """Download a Sentinel-2 product using CDSE API - matches ShallowLearn implementation"""
        try:
            from cdsetool.download import download_features
            from cdsetool.credentials import Credentials
        except ImportError:
            raise Exception("cdsetool not available. Cannot download Sentinel-2 products.")
        
        import os
        from pathlib import Path
        
        # Use stored credentials
        creds = Credentials(username=self.username, password=self.password)
        
        # The product metadata should contain download info
        if not product.metadata:
            raise Exception(f"No metadata available for product {product.product_id}")
        
        try:
            # Create features list with just this product
            features = [product.metadata]
            
            # Use cdsetool to download
            os.makedirs(output_dir, exist_ok=True)
            print(f"Downloading {product.product_id}...")
            
            # Download using cdsetool - matches existing ShallowLearn pattern
            # Use cdsetool credentials for authentication
            creds = Credentials(username=self.username, password=self.password)
            
            # Set up credentials globally (cdsetool pattern)
            import cdsetool
            cdsetool.credentials.default_credentials = creds
            
            download_features(features, output_dir)
            
            # Find the downloaded file
            downloaded_files = list(Path(output_dir).glob(f"{product.product_id}*"))
            if downloaded_files:
                file_path = str(downloaded_files[0])
                print(f"Downloaded {product.product_id}")
                return file_path
            else:
                # Check for .zip files that might match
                zip_files = list(Path(output_dir).glob("*.zip"))
                if zip_files:
                    # Return the most recent zip file as a fallback
                    latest_zip = max(zip_files, key=os.path.getctime)
                    return str(latest_zip)
                else:
                    raise Exception(f"Download completed but file not found in {output_dir}")
                    
        except Exception as e:
            raise Exception(f"Download failed for {product.product_id}: {str(e)}")


class UnifiedSatelliteAPI:
    """Unified interface for both Landsat and Sentinel-2 APIs"""
    
    def __init__(self):
        self.landsat_api = None
        self.sentinel2_api = None
    
    def _initialize_apis(self, satellites: List[str]):
        """Initialize only the requested satellite APIs"""
        if "landsat" in satellites and self.landsat_api is None:
            try:
                self.landsat_api = LandsatUSGSDownloader()
            except Exception as e:
                print(f"Warning: Could not initialize Landsat API: {e}")
        
        if "sentinel2" in satellites and self.sentinel2_api is None:
            try:
                self.sentinel2_api = Sentinel2CDSEDownloader()
            except Exception as e:
                print(f"Warning: Could not initialize Sentinel-2 API: {e}")
    
    def search(self, query: SatelliteQuery) -> List[SatelliteProduct]:
        """Search all requested satellites independently"""
        self._initialize_apis(query.satellites)
        all_products = []
        
        # Search Landsat independently
        if "landsat" in query.satellites and self.landsat_api is not None:
            try:
                landsat_products = self.landsat_api.search(query)
                all_products.extend(landsat_products)
                print(f"✓ Landsat API returned {len(landsat_products)} products")
            except Exception as e:
                print(f"✗ Landsat API failed: {e}")
        
        # Search Sentinel-2 independently  
        if "sentinel2" in query.satellites and self.sentinel2_api is not None:
            try:
                sentinel2_products = self.sentinel2_api.search(query)
                all_products.extend(sentinel2_products)
                print(f"✓ Sentinel-2 API returned {len(sentinel2_products)} products")
            except Exception as e:
                print(f"✗ Sentinel-2 API failed: {e}")
        
        return all_products
    
    def download(self, products: List[SatelliteProduct], output_dir: str, 
                max_concurrent: int = 3) -> Dict[str, str]:
        """Download satellite products to specified directory
        
        Args:
            products: List of SatelliteProduct objects to download
            output_dir: Directory to save downloaded files
            max_concurrent: Maximum concurrent downloads
            
        Returns:
            Dictionary mapping product_id to local file path (or error message)
        """
        import os
        from pathlib import Path
        import threading
        
        os.makedirs(output_dir, exist_ok=True)
        results = {}
        semaphore = threading.Semaphore(max_concurrent)
        threads = []
        
        def download_single(product: SatelliteProduct):
            semaphore.acquire()
            try:
                if product.satellite == "landsat" and self.landsat_api:
                    file_path = self.landsat_api.download_product(product, output_dir)
                    results[product.product_id] = file_path
                elif product.satellite == "sentinel2" and self.sentinel2_api:
                    file_path = self.sentinel2_api.download_product(product, output_dir)  
                    results[product.product_id] = file_path
                else:
                    results[product.product_id] = f"Error: No API available for {product.satellite}"
                    
            except Exception as e:
                results[product.product_id] = f"Error: {str(e)}"
            finally:
                semaphore.release()
        
        # Start download threads
        for product in products:
            thread = threading.Thread(target=download_single, args=(product,))
            threads.append(thread)
            thread.start()
        
        # Wait for all downloads to complete
        for thread in threads:
            thread.join()
            
        return results


def test_against_existing_parameters():
    """Test unified API using the same parameters as existing working files"""
    
    print("=== TESTING UNIFIED API AGAINST EXISTING PARAMETERS ===")
    
    api = UnifiedSatelliteAPI()
    
    # Test 1: Using BarAlHikman Landsat parameters
    print("\n1. Testing with BarAlHikman Landsat parameters...")
    landsat_query = SatelliteQuery(
        geometry=(58.3718, 20.3297, 58.4414, 20.3598),  # Same as spatialFilter in data_download.py
        date_range=("2023-02-01", "2024-03-15"),  # Recent subset of Run 2 dates
        cloud_cover_max=100,
        processing_level="L1",
        satellites=["landsat"],
        max_results=10  # Limit for testing
    )
    
    landsat_products = api.search(landsat_query)
    print(f"Unified API - Landsat results: {len(landsat_products)}")
    if landsat_products:
        print("Sample Landsat product:")
        print(f"  ID: {landsat_products[0].product_id}")
        print(f"  Sensor: {landsat_products[0].sensor}")
        print(f"  Date: {landsat_products[0].acquisition_date}")
        print(f"  Cloud Cover: {landsat_products[0].cloud_cover}%")
    
    # Test 2: Using ShallowLearn Sentinel-2 parameters
    print("\n2. Testing with ShallowLearn Sentinel-2 parameters...")
    sentinel2_query = SatelliteQuery(
        geometry=(145.1453, -15.4558, 146.1558, -14.4626),  # Same as bbox in DownloadData.py
        date_range=("2023-01-01", "2024-01-31"),  # Recent subset
        cloud_cover_max=100,
        processing_level="L1C", 
        satellites=["sentinel2"],
        max_results=10  # Limit for testing
    )
    
    sentinel2_products = api.search(sentinel2_query)
    print(f"Unified API - Sentinel-2 results: {len(sentinel2_products)}")
    if sentinel2_products:
        print("Sample Sentinel-2 product:")
        print(f"  ID: {sentinel2_products[0].product_id}")
        print(f"  Sensor: {sentinel2_products[0].sensor}")
        print(f"  Date: {sentinel2_products[0].acquisition_date}")
        print(f"  Cloud Cover: {sentinel2_products[0].cloud_cover}%")
    
    # Test 3: Combined query
    print("\n3. Testing combined Landsat + Sentinel-2 query...")
    combined_query = SatelliteQuery(
        geometry=(58.3718, 20.3297, 58.4414, 20.3598),  # Bar Al Hikman area
        date_range=("2023-06-01", "2023-08-31"),  # Summer 2023
        cloud_cover_max=30,
        satellites=["landsat", "sentinel2"],
        max_results=5  # Limit for testing
    )
    
    all_products = api.search(combined_query)
    landsat_count = len([p for p in all_products if p.satellite == 'landsat'])
    sentinel2_count = len([p for p in all_products if p.satellite == 'sentinel2'])
    
    print(f"Combined query results:")
    print(f"  Landsat: {landsat_count}")
    print(f"  Sentinel-2: {sentinel2_count}")
    print(f"  Total: {len(all_products)}")
    
    return all_products


if __name__ == "__main__":
    test_against_existing_parameters()