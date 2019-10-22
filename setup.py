#!/usr/bin/env python

from setuptools import setup
from setuptools import find_packages


def get_version():
    with open("gridded/__init__.py") as initfile:
        for line in initfile:
            if "__version__" in line:
                return line.split('=', 1)[1].strip().strip('"')
    return ""


# install_requires = [line.strip() for line in open('conda_requirements.txt')]
# tests_require = [line.strip() for line in open('conda_requirements_dev.txt')]

# leaving requirements out of setup.py, as pip  may not work right anyway.
# requirements should be pre-installed with conda (or maybe pip)
install_requires = []
tests_require = []

config = {'name': 'gridded',
          'description': "Unified API for working with results from (Met/Ocean) models on"
                         "various grid types",
          'long_description': open('README.rst').read(),
          'author': 'Chris Barker, Jay Hennen',
          'url': "https://github.com/NOAA-ORR-ERD/gridded",
          'download_url': "https://github.com/NOAA-ORR-ERD/gridded",
          'author_email': 'chris.barker@noaa.gov',
          'version': get_version(),
          'install_requires': install_requires,
          'tests_require': tests_require,
          'packages': find_packages(),
          'package_data': {'gridded': ['tests/test_data/*',
                                       'tests/test_ugrid/files/*'
                                       ]},
          'scripts': [],
          }

# Add in any extra build steps for cython, etc.

setup(**config)
