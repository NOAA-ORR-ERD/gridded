#!/usr/bin/env python

# __init__.py for pyugrid package
# This brings in the names we want in the package.

"""
pyugrid package

A Python API to utilize data written using the netCDF unstructured grid conventions:

https://github.com/ugrid-conventions/ugrid-conventions.

This package contains code for reading/writing netcdf files (and potentially other formats)
as well as code for working with data on unstructured grids

"""

from __future__ import (absolute_import, division, print_function)

from .ugrid import UGrid
from .uvar import UVar
from .uvar import UMVar
from . import grid_io

__version__ = '0.3.1'

__all__ = ['UGrid', 'UVar', 'UMVar', 'grid_io']
