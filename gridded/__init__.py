#!/usr/bin/env python

__version__ = "0.7.5"

VALID_SGRID_LOCATIONS = (None, "center", "edge1", "edge2", "node")
VALID_UGRID_LOCATIONS = (None, "node", "face", "edge", "boundary")
VALID_LOCATIONS = set(VALID_SGRID_LOCATIONS + VALID_UGRID_LOCATIONS)

from .depth import DepthBase, L_Depth, S_Depth
from .gridded import Dataset
from .grids import Grid, Grid_R, Grid_S, Grid_U
from .time import Time
from .variable import Variable, VectorVariable

DepthBase._default_component_types["variable"] = Variable
S_Depth._default_component_types["variable"] = Variable
S_Depth._default_component_types["bathymetry"] = Variable
S_Depth._default_component_types["zeta"] = Variable

from .time import OutOfTimeRangeError

__all__ = [
    "Variable",
    "VectorVariable",
    "Grid",
    "Grid_R",
    "Grid_S",
    "Grid_U",
    "Dataset",
    "Time",
    "OutOfTimeRangeError",
]
