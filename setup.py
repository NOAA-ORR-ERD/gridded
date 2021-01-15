#!/usr/bin/env python
import os
import os.path as osp

from setuptools import setup
from setuptools import find_packages
from setuptools import Extension


USE_CYTHON = os.getenv('USE_CYTHON', '').lower()
USE_CYTHON = USE_CYTHON and (not USE_CYTHON.startswith('f') and
                             not USE_CYTHON.startswith('n'))


ext = '.pyx' if USE_CYTHON else '.c'


extensions = [
    Extension(
        'gridded.pyugrid.inverse_lookup',
        sources=[osp.join("gridded", "pyugrid", "inverse_lookup" + ext)],
    )
]


if USE_CYTHON:
    from Cython.Build import cythonize
    extensions = cythonize(extensions)


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
          'ext_modules': extensions,
          }

# Add in any extra build steps for cython, etc.

setup(**config)
