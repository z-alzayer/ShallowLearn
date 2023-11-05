import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

def reproject_raster(input_raster_path: str, output_raster_path: str, dst_crs: str):
    """
    Reproject a raster file to a different CRS.
    
    Parameters:
    input_raster_path (str): Path to the input raster file.
    output_raster_path (str): Path where the reprojected raster will be saved.
    dst_crs (str): The target CRS in Proj4 or EPSG format.
    """
    
    with rasterio.open(input_raster_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height,
            'dtype': 'uint8',
            'compress': 'lzw'
        })
        
        with rasterio.open(output_raster_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs)
                
    print(f"Reprojected raster saved at: {output_raster_path}")


if __name__ == "__main__":
    reproject_raster("/home/zba21/Documents/ShallowLearn/Data/Allan_Coral_atlas_image.tif", "/home/zba21/Documents/ShallowLearn/Data/Allan_Coral_atlas_image_reprojected.tif", 32755)