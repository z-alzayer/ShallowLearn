[![codecov](https://codecov.io/github/z-alzayer/ShallowLearn/graph/badge.svg?token=TKUAGXODRB)](https://codecov.io/github/z-alzayer/ShallowLearn)

# ShallowLearn: A Python Toolkit for Shallow Water Remote Sensing Analysis

ShallowLearn provides a collection of tools and utilities designed for processing, analyzing, and visualizing satellite imagery, with a particular focus on shallow water environments like coral reefs. It includes functionalities for data loading, preprocessing, feature extraction, segmentation, time-series analysis, and visualization. This repo is a work in progress with improvements to come, I'm still in the process of refactoring now that the paper is published but hopefully I'll get to it sooner rather than later.

## Features

*   **Data Handling:**
    *   Load satellite data from various formats (GeoTIFF, Sentinel-2 SAFE/ZIP).
    *   Download Sentinel-2 via APIs (`cdsetool`).
    *   Compile multi-band GeoTIFFs from raw Sentinel-2 bands.
    *   Handle time series and seasonal data loading.
*   **Image Processing:**
    *   Radiometric normalization (PCA-based, Histogram Matching).
    *   Image transformations (Contrast Enhancement - LCE, BCET, Color Space Conversions - LAB, HSV).
    *   Cloud detection and masking (XGBoost-based).
    *   Image resampling operations.
*   **Feature Extraction:**
    *   Calculate standard remote sensing indices.
    *   Extract computer vision features (LBP, Gabor, HOG, Edge Density).
    *   Generate comprehensive feature stacks combining spectral, index, and texture information.
*   **Segmentation:**
    *   Superpixel generation using various algorithms (SLIC, Felzenszwalb, Quickshift, Watershed).
    *   Superpixel processing pipelines involving PCA and clustering (DBSCAN, OPTICS, GMM).
*   **Depth Invariant Indices (DII):**
    *   Calculate standard band-ratio DII.
    *   Implement superpixel-based DII workflows for automated deep/shallow area identification.
*   **Time Series Analysis:**
    *   Process and normalize image time series.
    *   Quick-look analysis using PCA and clustering on image thumbnails or cropped areas.
*   **Utilities & Visualization:**
    *   Helper functions for date extraction, band mapping, and metadata handling.
    *   Plotting utilities for images, spectra, histograms, scatter plots with image overlays, and density plots.


[Documentation is available here](https://z-alzayer.github.io/ShallowLearn/)

If you've found any of the code useful please cite our [paper](https://www.mdpi.com/2072-4292/17/7/1244) 
