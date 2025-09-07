"""
Real API tests for Unified Satellite API - tests actual API calls without downloads
Tests both Sentinel-2 and Landsat functionality with real data
"""
import os
import pytest
import pandas as pd
from datetime import datetime
from pathlib import Path

from ShallowLearn.api.unified_satellite_api import (
    UnifiedSatelliteAPI, 
    SatelliteQuery, 
    SatelliteProduct,
    LandsatUSGSDownloader,
    Sentinel2CDSEDownloader
)


class TestSatelliteQuery:
    """Test SatelliteQuery construction and validation"""
    
    def test_basic_query_creation(self):
        """Test basic query creation"""
        query = SatelliteQuery(
            geometry=(-157.8, 21.2, -157.7, 21.3),
            date_range=("2023-02-01", "2023-02-15"),
            cloud_cover_max=50,
            satellites=["sentinel2", "landsat"],
            max_results=10
        )
        
        assert query.geometry == (-157.8, 21.2, -157.7, 21.3)
        assert query.date_range == ("2023-02-01", "2023-02-15")
        assert query.cloud_cover_max == 50
        assert set(query.satellites) == {"sentinel2", "landsat"}
        assert query.max_results == 10
    
    def test_query_with_processing_level(self):
        """Test query with processing level specification"""
        query = SatelliteQuery(
            geometry=(-157.8, 21.2, -157.7, 21.3),
            date_range=("2023-02-01", "2023-02-15"),
            processing_level="L2"
        )
        
        assert query.processing_level == "L2"


class TestSatelliteProduct:
    """Test SatelliteProduct data structure"""
    
    def test_product_creation_landsat(self):
        """Test Landsat product creation"""
        product = SatelliteProduct(
            product_id="LC80640452023162LGN00",
            satellite="landsat",
            sensor="OLI/TIRS",
            acquisition_date="2023-06-11T00:00:00.000000Z",
            cloud_cover=15.0,
            processing_level="L1"
        )
        
        assert product.satellite == "landsat"
        assert product.sensor == "OLI/TIRS"
        assert product.cloud_cover == 15.0
    
    def test_product_creation_sentinel2(self):
        """Test Sentinel-2 product creation"""
        product = SatelliteProduct(
            product_id="S2A_MSIL1C_20230207T211921_N0510_R100_T04QFJ_20240723T061500.SAFE",
            satellite="sentinel2",
            sensor="MSI",
            acquisition_date="2023-02-07T21:19:21.024000Z",
            cloud_cover=12.5,
            processing_level="L1C",
            platform="Sentinel-2A"
        )
        
        assert product.satellite == "sentinel2"
        assert product.platform == "Sentinel-2A"
        assert product.processing_level == "L1C"
    
    def test_product_to_dict(self):
        """Test product serialization to dictionary"""
        product = SatelliteProduct(
            product_id="test_product",
            satellite="sentinel2",
            sensor="MSI",
            acquisition_date="2023-02-07T21:19:21.024000Z",
            cloud_cover=10.0,
            processing_level="L1C"
        )
        
        data = product.to_dict()
        assert isinstance(data, dict)
        assert data['product_id'] == "test_product"
        assert data['satellite'] == "sentinel2"
        assert data['cloud_cover'] == 10.0


@pytest.mark.skipif(
    not (os.getenv('LSAT_USER') and os.getenv('LSAT_TOKEN')),
    reason="Landsat credentials required"
)
class TestLandsatUSGSDownloader:
    """Test Landsat API functionality with real API calls"""
    
    def test_landsat_initialization(self):
        """Test Landsat API initialization"""
        api = LandsatUSGSDownloader()
        
        assert api.username is not None
        assert api.token is not None
        assert api.service_url == "https://m2m.cr.usgs.gov/api/api/json/stable/"
        assert 'landsat_ot_c2_l1' in api.datasets.values()
    
    def test_landsat_authentication(self):
        """Test Landsat authentication flow"""
        api = LandsatUSGSDownloader()
        
        # Test login
        payload = {'username': api.username, 'token': api.token}
        api_key = api._send_request("login-token", payload)
        
        assert api_key is not None
        assert len(api_key) > 0
        
        # Test logout
        api._send_request("logout", None, api_key)
    
    def test_landsat_search_real_data(self):
        """Test Landsat search with real data"""
        query = SatelliteQuery(
            geometry=(58.3718, 20.3297, 58.4414, 20.3598),  # BarAlHikman coordinates
            date_range=("2023-06-01", "2023-06-15"),
            cloud_cover_max=50,
            satellites=["landsat"],
            max_results=5
        )
        
        api = LandsatUSGSDownloader()
        products = api.search(query)
        
        assert len(products) > 0
        assert len(products) <= 5
        
        for product in products:
            assert product.satellite == "landsat"
            assert product.product_id is not None
            assert isinstance(product.cloud_cover, (int, float))
            assert product.acquisition_date is not None
            assert product.processing_level is not None
    
    def test_landsat_download_options(self):
        """Test getting download options for Landsat products"""
        query = SatelliteQuery(
            geometry=(58.3718, 20.3297, 58.4414, 20.3598),  # BarAlHikman coordinates
            date_range=("2023-06-01", "2023-06-15"),
            cloud_cover_max=30,
            satellites=["landsat"],
            max_results=1
        )
        
        api = LandsatUSGSDownloader()
        products = api.search(query)
        
        if products:
            # Test getting download options (without downloading)
            product = products[0]
            
            # Login
            payload = {'username': api.username, 'token': api.token}
            api_key = api._send_request("login-token", payload)
            
            try:
                # Get download options
                payload = {
                    'datasetName': api.datasets.get(product.processing_level, 'landsat_ot_c2_l1'),
                    'entityIds': [product.product_id]
                }
                
                download_options = api._send_request("download-options", payload, api_key)
                
                assert isinstance(download_options, list)
                assert len(download_options) > 0
                
                # Verify structure of download options
                for option in download_options:
                    assert 'available' in option
                    assert 'id' in option
                    assert 'productName' in option
                    assert isinstance(option['available'], bool)
                
                # Should have at least some available options
                available_options = [opt for opt in download_options if opt.get('available')]
                print(f"Found {len(available_options)} available download options for {product.product_id}")
                
            finally:
                # Logout
                api._send_request("logout", None, api_key)


@pytest.mark.skipif(
    not (os.getenv('SEN_USER') and os.getenv('SEN_PASS')),
    reason="Sentinel-2 credentials required"
)
class TestSentinel2CDSEDownloader:
    """Test Sentinel-2 API functionality with real API calls"""
    
    def test_sentinel2_initialization(self):
        """Test Sentinel-2 API initialization"""
        api = Sentinel2CDSEDownloader()
        
        assert api.username is not None
        assert api.password is not None
    
    def test_sentinel2_search_real_data(self):
        """Test Sentinel-2 search with real data"""
        query = SatelliteQuery(
            geometry=(58.3718, 20.3297, 58.4414, 20.3598),  # BarAlHikman coordinates
            date_range=("2023-02-01", "2023-02-15"),
            cloud_cover_max=50,
            satellites=["sentinel2"],
            max_results=5
        )
        
        api = Sentinel2CDSEDownloader()
        products = api.search(query)
        
        assert len(products) > 0
        assert len(products) <= 5
        
        for product in products:
            assert product.satellite == "sentinel2"
            assert product.product_id is not None
            assert isinstance(product.cloud_cover, (int, float))
            assert product.acquisition_date is not None
            assert product.processing_level is not None
            assert product.platform in ["Sentinel-2A", "Sentinel-2B"]
    
    def test_sentinel2_product_metadata(self):
        """Test Sentinel-2 product has expected metadata fields"""
        query = SatelliteQuery(
            geometry=(-157.8, 21.2, -157.7, 21.3),
            date_range=("2023-02-01", "2023-02-15"),
            cloud_cover_max=30,
            satellites=["sentinel2"],
            max_results=1
        )
        
        api = Sentinel2CDSEDownloader()
        products = api.search(query)
        
        if products:
            product = products[0]
            
            # Check Sentinel-2 specific fields
            assert hasattr(product, 'orbit_number')
            assert hasattr(product, 'relative_orbit_number')
            assert hasattr(product, 'processing_baseline')
            assert hasattr(product, 'timeliness')
            
            # These fields should be populated
            assert product.platform is not None
            assert product.instrument == "MSI"


class TestUnifiedSatelliteAPI:
    """Test unified API functionality with real data"""
    
    def test_unified_api_initialization(self):
        """Test unified API initialization"""
        api = UnifiedSatelliteAPI()
        
        assert api.landsat_api is None
        assert api.sentinel2_api is None
    
    @pytest.mark.skipif(
        not (os.getenv('SEN_USER') and os.getenv('SEN_PASS')),
        reason="Sentinel-2 credentials required"
    )
    def test_sentinel2_only_search(self):
        """Test search with only Sentinel-2"""
        query = SatelliteQuery(
            geometry=(-157.8, 21.2, -157.7, 21.3),
            date_range=("2023-02-01", "2023-02-15"),
            cloud_cover_max=50,
            satellites=["sentinel2"],
            max_results=3
        )
        
        api = UnifiedSatelliteAPI()
        products = api.search(query)
        
        assert len(products) > 0
        assert all(p.satellite == "sentinel2" for p in products)
        assert api.sentinel2_api is not None
        assert api.landsat_api is None
    
    @pytest.mark.skipif(
        not (os.getenv('LSAT_USER') and os.getenv('LSAT_TOKEN')),
        reason="Landsat credentials required"
    )
    def test_landsat_only_search(self):
        """Test search with only Landsat"""
        query = SatelliteQuery(
            geometry=(-157.8, 21.2, -157.7, 21.3),
            date_range=("2023-06-01", "2023-06-15"),
            cloud_cover_max=50,
            satellites=["landsat"],
            max_results=3
        )
        
        api = UnifiedSatelliteAPI()
        products = api.search(query)
        
        assert len(products) > 0
        assert all(p.satellite == "landsat" for p in products)
        assert api.landsat_api is not None
        assert api.sentinel2_api is None
    
    @pytest.mark.skipif(
        not (os.getenv('SEN_USER') and os.getenv('SEN_PASS') and 
             os.getenv('LSAT_USER') and os.getenv('LSAT_TOKEN')),
        reason="Both Sentinel-2 and Landsat credentials required"
    )
    def test_mixed_satellite_search(self):
        """Test search with both satellites"""
        query = SatelliteQuery(
            geometry=(-157.8, 21.2, -157.7, 21.3),
            date_range=("2023-02-01", "2023-06-15"),
            cloud_cover_max=50,
            satellites=["sentinel2", "landsat"],
            max_results=10
        )
        
        api = UnifiedSatelliteAPI()
        products = api.search(query)
        
        assert len(products) > 0
        
        satellites = {p.satellite for p in products}
        assert len(satellites) > 0  # Should have at least one type
        
        # Check that products are properly typed
        sentinel_products = [p for p in products if p.satellite == "sentinel2"]
        landsat_products = [p for p in products if p.satellite == "landsat"]
        
        for p in sentinel_products:
            assert p.platform in ["Sentinel-2A", "Sentinel-2B"]
            assert p.sensor == "MSI"
        
        for p in landsat_products:
            assert p.sensor in ["OLI/TIRS", "OLI", "TM", "ETM+"]
        
        assert api.landsat_api is not None
        assert api.sentinel2_api is not None
    
    def test_products_to_dataframe(self):
        """Test conversion to DataFrame"""
        # Create some test products
        products = [
            SatelliteProduct(
                product_id="S2A_MSIL1C_20230207T211921_N0510_R100_T04QFJ_20240723T061500.SAFE",
                satellite="sentinel2",
                sensor="MSI",
                acquisition_date="2023-02-07T21:19:21.024000Z",
                cloud_cover=12.5,
                processing_level="L1C",
                platform="Sentinel-2A"
            ),
            SatelliteProduct(
                product_id="LC80640452023162LGN00",
                satellite="landsat",
                sensor="OLI/TIRS",
                acquisition_date="2023-06-11T00:00:00.000000Z",
                cloud_cover=15.0,
                processing_level="L1"
            )
        ]
        
        api = UnifiedSatelliteAPI()
        df = api.products_to_dataframe(products)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert 'product_id' in df.columns
        assert 'satellite' in df.columns
        assert 'acquisition_date' in df.columns
        assert 'cloud_cover' in df.columns
        
        # Check date conversion
        assert pd.api.types.is_datetime64_any_dtype(df['acquisition_date'])
        assert not df['acquisition_date'].isna().any()
        
        # Check sorting
        assert df['acquisition_date'].is_monotonic_increasing
    
    def test_create_download_manifest(self):
        """Test download manifest creation"""
        products = [
            SatelliteProduct(
                product_id="test_product_1",
                satellite="sentinel2",
                sensor="MSI",
                acquisition_date="2023-02-07T21:19:21.024000Z",
                cloud_cover=12.5,
                processing_level="L1C"
            )
        ]
        
        api = UnifiedSatelliteAPI()
        
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = api.create_download_manifest(products, str(Path(temp_dir) / "test_manifest.csv"))
            
            assert Path(manifest_path).exists()
            
            # Read and verify manifest
            manifest_df = pd.read_csv(manifest_path)
            assert len(manifest_df) == 1
            assert 'product_id' in manifest_df.columns
            assert 'satellite' in manifest_df.columns
            assert manifest_df.iloc[0]['product_id'] == "test_product_1"


if __name__ == "__main__":
    pytest.main([__file__])