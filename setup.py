from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))


VERSION = '0.0.1'
DESCRIPTION = 'Shallow water remote sensing package'
LONG_DESCRIPTION = 'Package that integrates remote sensing workflows for shallow water environments and machine learning'

# Setting up
setup(
    name="ShallowLearn",
    version=VERSION,
    author="z-z",
    author_email="<na@protonmail.com>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    package_data={'': ['Data/Clear_Reefs.csv']},
    install_requires=['numpy','matplotlib','pandas'],
    keywords=['remote sensing etc'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)