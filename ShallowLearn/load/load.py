import math
import os
import tarfile

from osgeo import gdal
from pyproj import CRS, Transformer


def parse_mtl_from_tar(tar_path):
    """Parse the Landsat MTL.txt file inside a tar archive into a flat dictionary."""
    mtl_dict = {}
    stack = []
    with tarfile.open(tar_path, "r") as tar:
        mtl_member = next(
            (m for m in tar.getmembers() if m.name.endswith("MTL.txt")), None
        )
        if mtl_member is None:
            raise FileNotFoundError("MTL.txt not found in tar archive")
        mtl_file = tar.extractfile(mtl_member)
        for line_bytes in mtl_file:
            line = line_bytes.decode("utf-8").strip()
            if not line or line == "END":
                continue
            if line.startswith("GROUP = "):
                group = line.split("=", 1)[1].strip()
                stack.append(group)
            elif line.startswith("END_GROUP"):
                stack.pop()
            elif "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"')
                full_key = ".".join(stack + [key])
                mtl_dict[full_key] = value
    return mtl_dict


def expand_bounds(bounds, n_pixels=10, pixel_size=30):
    """Expand bounds (in lon/lat) by n_pixels * pixel_size meters in all directions."""
    minx, miny, maxx, maxy = bounds
    mean_lat = (miny + maxy) / 2
    expand_m = n_pixels * pixel_size
    meters_per_degree_lat = 111320
    meters_per_degree_lon = 111320 * math.cos(math.radians(mean_lat))
    expand_deg_lat = expand_m / meters_per_degree_lat
    expand_deg_lon = expand_m / meters_per_degree_lon
    return (
        minx - expand_deg_lon,
        miny - expand_deg_lat,
        maxx + expand_deg_lon,
        maxy + expand_deg_lat,
    )


def get_raster_crs_from_tar(tar_path, band_file):
    """Get the CRS of a raster band inside a tar archive."""
    vsi_path = f"/vsitar/{tar_path}/{band_file}"
    src_ds = gdal.Open(vsi_path)
    raster_crs_wkt = src_ds.GetProjection()
    src_ds = None
    return CRS.from_wkt(raster_crs_wkt)


def transform_bounds(bounds, src_crs, dst_crs):
    """Transform bounds from src_crs to dst_crs."""
    minx, miny, maxx, maxy = bounds
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    ulx, uly = transformer.transform(minx, maxy)
    lrx, lry = transformer.transform(maxx, miny)
    return ulx, uly, lrx, lry


def get_band_files_from_tar(tar_path):
    """Return sorted list of .TIF band files inside the tar."""
    with tarfile.open(tar_path, "r") as tar:
        members = tar.getnames()
    band_files = sorted([name for name in members if name.endswith(".TIF")])
    return band_files


def build_and_crop_vrt(
    tar_path,
    gpd_df,
    out_dir,
    n_pixels=10,
    pixel_size=30,
    project_name="Landsat Cropping Example",
):
    # 1. Expand bounds
    expanded_bounds = expand_bounds(gpd_df.total_bounds, n_pixels, pixel_size)

    # 2. Get band files and CRS
    band_files = get_band_files_from_tar(tar_path)
    if not band_files:
        raise RuntimeError(f"No .TIF files found in {tar_path}")
    raster_crs = get_raster_crs_from_tar(tar_path, band_files[0])

    # 3. Transform expanded bounds to raster CRS
    ulx, uly, lrx, lry = transform_bounds(
        expanded_bounds, "EPSG:4326", raster_crs.to_string()
    )

    # 4. Build /vsitar/ paths for all bands
    vsi_paths = [f"/vsitar/{tar_path}/{band_file}" for band_file in band_files]

    # 5. Build VRT
    vrt_base = os.path.splitext(os.path.basename(tar_path))[0]
    vrt_path = os.path.join(out_dir, f"{vrt_base}.vrt")
    vrt_ds = gdal.BuildVRT(vrt_path, vsi_paths, separate=True)
    vrt_ds = None

    # 6. Crop VRT
    cropped_vrt_path = os.path.join(out_dir, f"{vrt_base}_cropped.vrt")
    gdal.Translate(
        cropped_vrt_path, vrt_path, projWin=[ulx, uly, lrx, lry], format="VRT"
    )

    # 7. Parse MTL metadata
    mtl_metadata = parse_mtl_from_tar(tar_path)

    # 8. Add metadata to cropped VRT
    vrt_ds = gdal.Open(cropped_vrt_path, gdal.GA_Update)
    vrt_ds.SetMetadata(mtl_metadata, "MTL")
    metadata_dict = {
        "project": project_name,
        "source_tar": tar_path,
        "descriptions": ", ".join([i.split(".")[0].split("_")[-1] for i in band_files]),
        "expanded_bbox_wgs84": ",".join(map(str, expanded_bounds)),
        "expanded_bbox_proj": f"{ulx},{uly},{lrx},{lry}",
    }
    vrt_ds.SetMetadata(metadata_dict)
    band_descriptions = [i.split(".")[0].split("_")[-1] for i in band_files]
    for i, description in enumerate(band_descriptions, 1):
        if i <= vrt_ds.RasterCount:
            band = vrt_ds.GetRasterBand(i)
            band.SetDescription(description)
            band.SetMetadataItem("DESCRIPTION", description)
    vrt_ds = None
    print(f"Created: {cropped_vrt_path}")


def batch_process_landsat_tars(tar_list, gpd_df, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for tar_path in tar_list:
        try:
            build_and_crop_vrt(tar_path, gpd_df, out_dir)
        except Exception as e:
            print(f"Failed for {tar_path}: {e}")


if __name__ == "__main__":
    import geopandas as gpd

    import ShallowLearn.FileProcessing as fp

    # Example usage:
    gpd_df = gpd.read_file(
        "/home/zba21/Documents/Github/BarAlHikman/Notebooks/BarHikman.gpkg"
    )
    tar_list = [
        i
        for i in fp.list_files_in_dir_recur("/mnt/sda_mount/Landsat_CH3/")
        if i.endswith(".tar")
    ]
    out_dir = "/home/zba21/Documents/Github/BarAlHikman/data/"

    # Uncomment and set your variables:
    batch_process_landsat_tars(tar_list, gpd_df, out_dir)
    pass
