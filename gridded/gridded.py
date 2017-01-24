#!/usr/bin/env python

# py2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals

import copy

import numpy as np
import netCDF4 as nc4
from . import pysgrid
from . import pyugrid
from .grids import Grid
from .variable import Variable

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
        :type ncfile: filename of netcdf file or opendap url or open netCDF4 Dataset object
                     (could be other file types in the future)

        :param grid: a dataset.Grid object or anything that presents the same API.

        :param variables: a dict of dataset.Variable objects -- or anything that
                          presents the same API.

        Either a filename or grid and variable objects should be provided -- not both.
        """
        if ncfile is not None:
            self.nc_dataset = get_dataset(ncfile)
            self.filename = self.nc_dataset.filepath()
            # self.grid = pyugrid.UGrid.from_nc_dataset(ds)
            self.grid = Grid.from_netCDF(filename=self.filename, dataset=self.nc_dataset)
            # var_names = pyugrid.read_netcdf.find_variables(self.nc_dataset,
            #                                                self.grid.mesh_name)
            self.variables = self._load_variables(self.nc_dataset)
        else:  # no file passed in -- create from grid and variables
            self.filename = None
            self.grid = grid
            self.variables = variables

    def _load_variables(self, ds):
        # fixme: need a way to do this for non-compliant files

        variables = {}
        for k in ds.variables.keys():
            is_not_grid_attr = all([k not in str(v).split() for v in self.grid.grid_topology.values()])
            if is_not_grid_attr and self.grid.infer_location(ds[k]) is not None:
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

    def load_from_varnames(self, ncfile, topology):
        """
        Load a Gridded dataset by specifying the variable names used for the topology

        :param ncfile: a file to load the Dataset from.
        :type ncfile: filename of netcdf file or opendap url or open netCDF4 Dataset object
                     (could be other file types in the future)

        :param topology: variables that define the topology
        :type topology: dict of topology_role keys, and variable name values

        Docs about what is required for each grid type here.

        NOTE: the grid type will be inferered by what topology is provided.
        """

        raise NotImplementedError

    def save(self, filename, format='netcdf4'):
        """
        save the dataset to a file

        :param filename: full path to file to save to.

        :param format: format to save -- 'netcdf3' or 'netcdf4'
                       are the only options at this point.
        """
        raise NotImplementedError
