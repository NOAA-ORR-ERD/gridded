#!/usr/bin/env python

# py2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals

import copy

import numpy as np
import netCDF4 as nc4
from . import pysgrid
from . import pyugrid
from .grids import Grid

from .utilities import asarraylike, get_dataset


"""
The main gridded.Dataset code
"""

import netCDF4


class Dataset():
    """
    An object that represent an entire complete dataset -- a collection of Variable,
    and the grid that they are stored on.
    """

    def __init__(self, ncfile=None, grid=None, variables=None, grid_topology=None,):
        """
        Construct a gridded.Dataset object. Can be constructed from a data file,
        or also raw grid and variable objects.

        :param ncfile: a file to load the Dataset from.
        :type param: filename of netcdf file or opendap url or open netCDF4 Dataset object
                     (could be other file types in the future)

        :param grid: a dataset.Grid object or anything that presents the same API.

        :param variables: a dict of dataset.Variable objects -- or anything that
                          presents the same API.

        Either a filename or grid and variable objects should be provided -- not both.
        """
        if ncfile is not None:
            self.nc_dataset = get_dataset(ncfile)
            self.filename = self.nc_dataset.filepath()
            self.grid = Grid.from_netCDF(filename=self.filename, dataset=self.nc_dataset)
            self.variables = self._load_variables(self.nc_dataset)
        else:  # no file passed in -- create from grid and variables
            self.filename = None
            self.grid = grid
            self.variables = variables

    def _load_grid(self, ds):
        """
        load a grid from an open netCDF4 Dataset
        """
        # fixme: we may want to move the "magic" into here, rther than the GRid constructor
        # # try to load it as a compliant UGRID

        grid = pyugrid.UGrid.from_nc_dataset(ds)
        # grid = Grid.from_netCDF(dataset=ds,
        #                         # grid_topology=grid_topology  # where is this supposed to come from?
        #                         )

        print("loaded the grid:", grid)
        return grid

    def _load_variables(self, ds):
        # fixme: need a way to do this for non-compliant files

        variables = {}
        for k in ds.variables.keys():
            # check if the variable is a grid attribute
            is_not_grid_attr = all([k not in str(v).split() for v in self.grid.grid_topology.values()])
            shape_is_compatible = self.grid.infer_location(ds[k]) is not None
            if is_not_grid_attr and shape_is_compatible:
                try:
                    ln = ds[k].long_name
                except:
                    ln = ds[k].name
                variables[k] = Variable.from_netCDF(dataset=ds,
                                                    name=ln,
                                                    varname=k,
                                                    grid=self.grid,
                                                    )

        return variables

from .variable import Variable
