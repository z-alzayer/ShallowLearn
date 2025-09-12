import re
from pathlib import Path

import numpy as np
import rasterio as rio
from PIL import Image


class LandSatImage:
    """Landsat image with strict band ordering and missing band handling"""

    # Canonical band order with index mapping
    band_order = {
        "B1": 0,
        "B2": 1,
        "B3": 2,
        "B4": 3,
        "B5": 4,
        "B6": 5,
        "B7": 6,
        "B8": 7,
        "B9": 8,
        "B10": 9,
        "B11": 10,
        "SAA": 11,
        "SZA": 12,
        "VAA": 13,
        "VZA": 14,
        "PIXEL": 15,
        "RADSAT": 16,
    }

    def __init__(self, file_path):
        self.path = Path(file_path)
        self.meta = {}
        self.mtl_tags = {}
        self.tags = {}
        self.present_bands = set()

        with rio.open(self.path) as src:
            # Store metadata
            self.meta = src.meta.copy()
            self.tags = src.tags()
            self.mtl_tags = src.tags(ns="MTL")

            # Read all available bands
            band_data = {}
            for i in range(src.count):
                band_desc = src.descriptions[i]
                if band_desc in self.band_order:
                    band_data[band_desc] = src.read(i + 1)
                    self.present_bands.add(band_desc)

            # Create ordered array with placeholders
            ordered_bands = []
            self.band_status = {}

            for band_name in sorted(self.band_order, key=lambda x: self.band_order[x]):
                if band_name in band_data:
                    ordered_bands.append(band_data[band_name])
                    self.band_status[band_name] = True
                else:
                    # Create NaN placeholder with same shape/dtype as first band
                    if band_data:
                        placeholder = np.full_like(
                            next(iter(band_data.values())), np.nan
                        )
                    else:
                        placeholder = np.empty(
                            (self.meta["height"], self.meta["width"]), dtype="float32"
                        )
                        placeholder[:] = np.nan
                    ordered_bands.append(placeholder)
                    self.band_status[band_name] = False

            self.image = np.stack(ordered_bands, axis=0)
            self.image = np.transpose(self.image, (1, 2, 0))  # (x, y, channels)

    def __repr__(self):
        band_list = [
            f"{b} {'✓' if self.band_status[b] else '✗'}"
            for b in sorted(self.band_order, key=lambda x: self.band_order[x])
        ]
        return (
            f"<LandSatImage: {self.path}\n"
            f"  Bands: {band_list}\n"
            f"  Shape: {self.image.shape}\n"
            f"  Missing: {sum(not v for v in self.band_status.values())} bands>"
        )


class LandSatImagery:
    """Managed collection with date sorting and strict band order"""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.image_files = self._sorted_image_files()
        self.images = [LandSatImage(f) for f in self.image_files]

    def _sorted_image_files(self):
        def extract_date(filename):
            parts = filename.stem.split("_")
            if len(parts) > 3 and re.match(r"\d{8}", parts[3]):
                return parts[3]
            return ""

        files = [f for f in self.directory.glob("*cropped.vrt") if "LE07" not in f.name]
        return sorted(files, key=extract_date)

    def __iter__(self):
        return iter(self.images)

    def __getitem__(self, index):
        return self.images[index]

    def __len__(self):
        return len(self.images)

    def __repr__(self):
        return f"<LandSatImagery count={len(self)}>"

    def common_bands(self):
        """Get bands present in ALL images"""
        if not self.images:
            return set()
        band_order = {
            "B1": 0,
            "B2": 1,
            "B3": 2,
            "B4": 3,
            "B5": 4,
            "B6": 5,
            "B7": 6,
            "B8": 7,
            "B9": 8,
            "B10": 9,
            "B11": 10,
        }

        # Collect all band presence sets
        band_sets = [img.present_bands for img in self.images]
        # Find intersection across all images
        common_bands = set.intersection(*band_sets) if band_sets else set()

        # Restrict to only bands present in band_order
        common_bands_filtered = common_bands.intersection(band_order.keys())
        # Find intersection across all images
        return common_bands_filtered

    def get_common_bands_array(self):
        common_bands = self.common_bands()
        if not common_bands or not self.images:
            return np.array([])

        # Get canonical band indices for common bands
        band_indices = sorted(
            [LandSatImage.band_order[b] for b in common_bands], key=lambda x: x
        )

        # Find maximum spatial dimensions
        max_height = max(img.image.shape[0] for img in self.images)
        max_width = max(img.image.shape[1] for img in self.images)

        # Resize and stack images
        resized_images = []
        for img in self.images:
            # Resize each band channel
            resized_bands = []
            for channel in range(img.image.shape[2]):
                band = img.image[:, :, channel]

                # Convert to PIL Image and resize
                pil_band = Image.fromarray(band)
                resized_band = pil_band.resize((max_width, max_height), Image.BILINEAR)

                resized_bands.append(np.array(resized_band))

            # Stack resized bands and select common channels
            resized_img = np.stack(resized_bands, axis=2)
            resized_images.append(resized_img[:, :, band_indices])

        return np.stack(resized_images, axis=0)

    def common_bands(self):
        """Get set of bands present in ALL images"""
        if not self.images:
            return set()
        return set.intersection(*[img.present_bands for img in self.images])
