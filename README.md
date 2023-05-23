# ShallowLearn: Shallow Water Imaging in Python

## Overview

ShallowLearn is a Python package for processing and analyzing imagery data of shallow water environments. This package is designed to provide an efficient and effective way to calculate common indices used in marine and environmental science, as well as having a basic toolbox for datascience applications for Shallow Water Imaging. Indices included Chlorophyll Index (CI), Ocean Color Index (OCI), Suspended Sediment Index (SSI), Turbidity Index (TI), Water Quality Index (WQI), and Normalized Difference Chlorophyll Index (NDCI).


## Getting Started

### Dependencies

ShallowLearn uses the conda package manager to handle its environment. All necessary packages are listed in the `environment.yml` file. The key dependencies include `numpy`, `matplotlib`, and `ipywidgets`.

### Installation

To install the ShallowLearn package and its dependencies, clone the repository and use conda to create the environment:

```bash
git clone https://github.com/yourusername/ShallowLearn.git
cd ShallowLearn
conda env create -f environment.yml
```

### Usage

ShallowLearn is a Python package and can be imported like any other package. A typical usage scenario would be as follows:
```
python

import numpy as np
from ShallowLearn.LoadData import LoadGeoTIFF
from ShallowLearn.LoadData import LoadFromCSV
from ShallowLearn.Indices import ci, oci, ssi, ti, wqi, ndci

data_source = '/path/to/your/data/multiband_raster.tif'
img = LoadGeoTIFF(data_source).load()

# or if you have a csv with a column called full_path (containing path to directory)
data_source = '/path/to/your/data/file.csv'
img_array = LoadFromCSV(data_source).load()

plt.imshow(ci(img))

# or 
for img in img_array:
  plt.imshow(img)

# Additional functionality is included in the jupyter notebooks 
'''
