#!/usr/bin/env python

from __future__ import (absolute_import,
                        division,
                        print_function,
                        unicode_literals)

__version__ = "0.2.4"

VALID_SGRID_LOCATIONS = (None, 'center','edge1','edge2','node')
VALID_UGRID_LOCATIONS = (None, 'node', 'face', 'edge', 'boundary')
VALID_LOCATIONS = set(VALID_SGRID_LOCATIONS + VALID_UGRID_LOCATIONS)


# This is a lot of recursive importing :-(

from gridded.gridded import Dataset
from gridded.grids import Grid
from gridded.variable import Variable, VectorVariable
from gridded.time import Time

from gridded.depth import DepthBase
DepthBase._default_component_types['variable'] = Variable



__all__ = ["Variable",
           "VectorVariable",
           "Grid",
           "Dataset",
           "Time"]
