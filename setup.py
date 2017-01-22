
from setuptools import setup

def get_version():
    with open("gridded/__init__.py") as initfile:
        for line in initfile:
            if "__version__" in line:
                return line.split('=', 1)[1].strip().strip('"')
    return ""


config = {
        'name': 'gridded',
        'description': open('README.rst').read(),
        'author': 'Chris Barker, Jay Hennen',
        'url': "https://github.com/NOAA-ORR-ERD/gridded",
        'download_url': "https://github.com/NOAA-ORR-ERD/gridded",
        'author_email': 'chris.barker@noaa.gov',
        'version': get_version(),
        'install_requires': [],
        'packages': ['gridded'],
        'scripts': [],
}

# Add in any extra build steps for cython, etc.

setup(**config)
