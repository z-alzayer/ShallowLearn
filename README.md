# ShallowLearn — unsupervised methods for shallow-water remote sensing

We've simplified this repo in order to enhance readability and accessibility, its contents are all in 3 notebooks with the data required included. Everything should be relatively straightforward to run, if there is anything feel free to let me know by raising an issue. If you used the previous packages and setup, you'll need to go back to the prior commits and I would suggest forking from there.

| | | |
|---|---|---|
| [**Finding the usable images in an archive**](notebooks/01_pca_image_selection.ipynb) | Which of 536 archived Sentinel-2 scenes are worth downloading? Flatten each whole image into one vector, run PCA over the archive, cluster with DBSCAN. Clear sky falls out as the darkest dense cluster. | ~5 s |
| [**Masking cloud and land over water**](notebooks/02_cloud_and_land_masking.ipynb) | Water absorbs infrared and clouds do not, so regress infrared response from the visible bands and threshold the prediction. No labelled cloud masks, and unlike a brightness cut it leaves the reef flat standing. Read through its residual, the same fit is an anomaly detector: on a "cloud-free" scene it finds the breaking surf and wisps of haze no threshold would catch. | ~20 s |
| [**Removing the depth signal**](notebooks/03_superpixel_depth_invariant_index.ipynb) | Lyzenga's depth-invariant index traditionally needs two hand-drawn reference polygons. SLIC superpixels plus PCA and clustering find both, so the result is reproducible from the image alone. | ~30 s |


## Running them

```bash
git clone https://github.com/z-alzayer/ShallowLearn && cd ShallowLearn
uv sync && uv run jupyter lab
```

That is the whole setup. The data is in the repository, so there is nothing to download and
no credentials to configure — clone, sync, run. `uv.lock` pins the exact package versions
the committed outputs were produced with.

Without `uv`: `pip install -r <(uv export --no-hashes)` , or just
`pip install numpy scipy scikit-learn scikit-image xgboost matplotlib pillow jupyter`.

## What is in here

```
notebooks/   the three tutorials, with outputs committed
data/        the three arrays they use, and the script that built them
```

`data/thumbnails_55LCD.npy` — 536 ESA Preview Images for Sentinel-2 tile 55LCD (Great
Barrier Reef, 2015–2024), the free ~25 kB quicklooks ESA ships with every product,
downsampled from 343×343 to 128×128 so the archive fits in git.

`data/reef_cloudy.npy` — one reef clipped from a 55LCD L1C scene (2016-03-23), 604×598 at
10 m, all 13 MSI bands as delivered (DN = reflectance × 10 000). Broken cumulus over its
northern third is what makes it useful: the masking notebook needs cloud to mask.

`data/reef_clear.npy` — a ribbon reef on the outer Great Barrier Reef, same tile,
2020-05-31, 656×333, same 13-band layout. Chosen for the opposite reason: not one pixel
crosses the cloud threshold, so the depth machinery can be seen on its own.


## Citation

Al Zayer, Z., Mason, P., Platt, R., John, C.M. (2025). *An Improved Machine Learning-Based
Method for Unsupervised Characterisation for Coral Reef Monitoring in Earth Observation
Time-Series Data.* Remote Sensing 17(7), 1244.
[mdpi.com/2072-4292/17/7/1244](https://www.mdpi.com/2072-4292/17/7/1244)

MIT licensed.
