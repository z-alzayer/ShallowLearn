# Installation

Installing **ShallowLearn** is straightforward. Follow these steps to get started:

## Prerequisites

Make sure you have Python 3.10.14 installed on your system. We recommend setting up a conda environment before installation. A yml enviornment file is provided. 

## Installing the Package

Clone the repository and install the package using pip:

```bash
git clone https://github.com/z-alzayer/ShallowLearn.git
cd ShallowLearn
conda env create -f environment.yml # (1)!
pip install -e .
```

1. If the environment fails due to dependancies in the future, recreating it is fairly straightforward for the methods, the primary dependancies are sklearn, pandas and  numpy.

!!! warning 
This has been tested primarily on linux environments, whilst it should work on windows there are no guarantees
