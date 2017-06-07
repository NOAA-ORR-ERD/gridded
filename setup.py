
from setuptools import setup
from setuptools import find_packages


def get_version():
    with open("gridded/__init__.py") as initfile:
        for line in initfile:
            if "__version__" in line:
                return line.split('=', 1)[1].strip().strip('"')
    return ""


config = {'name': 'gridded',
          'description': "Unified API for working with results from (Met/Ocean) models on"
                         "various grid types",
          'long_description': open('README.rst').read(),
          'author': 'Chris Barker, Jay Hennen',
          'url': "https://github.com/NOAA-ORR-ERD/gridded",
          'download_url': "https://github.com/NOAA-ORR-ERD/gridded",
          'author_email': 'chris.barker@noaa.gov',
          'version': get_version(),
          'install_requires': [],
          'packages': find_packages(),
          'scripts': [],
          }

# Add in any extra build steps for cython, etc.

setup(**config)
