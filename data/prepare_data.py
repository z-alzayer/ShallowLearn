"""Build the three arrays the notebooks use, from raw Sentinel-2 products.

You do not need to run this to follow the notebooks -- all three arrays are
already in this directory. It is here so their provenance is explicit and the
selection reproducible.

    python data/prepare_data.py --thumbnails /path/to/pvi_jpgs \
                                --cloudy-reef-tiff /path/to/76_<scene>.tiff \
                                --clear-reef-tiff  /path/to/5_<scene>.tiff

thumbnails_55LCD.npy, uint8 (536, 128, 128, 3)
    ESA Preview Images (PVI) for Sentinel-2 tile 55LCD (Great Barrier Reef,
    2015-2024). ESA ships these as 343x343x3 JPEGs at ~320 m/pixel: subsampled
    B04/B03/B02 as true colour. They are downsampled to 128x128 here purely so
    the archive fits in a git repository (343x343 is 189 MB raw). The principal
    component structure is unchanged by this -- only the scale of the PC axes
    changes, since Euclidean distance in pixel space grows with sqrt(d).

reef_cloudy.npy, uint16 (604, 598, 13)
    One reef clipped from a Sentinel-2 L1C scene (55LCD, 2016-03-23), all 13
    MSI bands in native order B01..B12, quantised as ESA delivers them: DN =
    reflectance * 10000. Chosen for its broken cumulus: the cloud masking
    notebook needs cloud to mask.

reef_clear.npy, uint16 (656, 333, 13)
    A ribbon reef on the outer Great Barrier Reef, same tile, 2020-05-31, same
    13-band layout. Chosen for the opposite reason: not one pixel crosses the
    cloud mask threshold, so the depth-invariant index can be demonstrated
    without cloud confusing the picture.
"""

import argparse
import glob
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
THUMB_SIZE = 128


def build_thumbnails(src_dir, out_path):
    from PIL import Image

    files = sorted(glob.glob(os.path.join(src_dir, "*.jpg")))
    if not files:
        raise SystemExit(f"no .jpg thumbnails under {src_dir}")

    # A handful of products ship a full-resolution 2422px PVI instead of the
    # standard 343px one; resizing everything to a common size absorbs that.
    stack = np.stack(
        [
            np.array(Image.open(f).convert("RGB").resize((THUMB_SIZE, THUMB_SIZE)))
            for f in files
        ]
    )
    np.save(out_path, stack)

    names = [os.path.basename(f) for f in files]
    with open(os.path.join(HERE, "thumbnails_55LCD_filenames.txt"), "w") as fh:
        fh.write("\n".join(names) + "\n")

    print(f"{out_path}: {stack.shape} {stack.dtype} {stack.nbytes / 1e6:.1f} MB")


def build_reef(src_tiff, out_path):
    import rasterio

    with rasterio.open(src_tiff) as src:
        cube = src.read().transpose(1, 2, 0)  # (band, y, x) -> (y, x, band)

    if cube.shape[2] != 13:
        raise SystemExit(f"expected 13 bands, got {cube.shape[2]}")

    np.save(out_path, cube)
    print(f"{out_path}: {cube.shape} {cube.dtype} {cube.nbytes / 1e6:.1f} MB")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--thumbnails", help="directory of ESA PVI .jpg files")
    ap.add_argument("--cloudy-reef-tiff", help="13-band GeoTIFF, a reef under broken cloud")
    ap.add_argument("--clear-reef-tiff", help="13-band GeoTIFF, a cloud-free reef")
    args = ap.parse_args()

    if args.thumbnails:
        build_thumbnails(args.thumbnails, os.path.join(HERE, "thumbnails_55LCD.npy"))
    if args.cloudy_reef_tiff:
        build_reef(args.cloudy_reef_tiff, os.path.join(HERE, "reef_cloudy.npy"))
    if args.clear_reef_tiff:
        build_reef(args.clear_reef_tiff, os.path.join(HERE, "reef_clear.npy"))
    if not (args.thumbnails or args.cloudy_reef_tiff or args.clear_reef_tiff):
        ap.print_help()


if __name__ == "__main__":
    main()
