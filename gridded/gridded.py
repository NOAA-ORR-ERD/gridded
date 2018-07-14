#!/usr/bin/env python

# py2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals

import copy

import numpy as np
import netCDF4 as nc4
from gridded.grids import Grid
from gridded.variable import Variable

from gridded.utilities import asarraylike, get_dataset


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

        :param ncfile: A file or files to load the Dataset from.
        :type ncfile: Can be one of:
                      - file path of netcdf file as a string
                      - opendap url
                      - list of file paths (uses a netCDF4 MFDataset)
                      - open netCDF4 Dataset object
                     (could be other file types in the future)

        :param grid: a dataset.Grid object or anything that presents the same API.

        :param variables: a dict of dataset.Variable objects -- or anything that
                          presents the same API.

        :param grid_topology: mapping of grid topology components to netcdf variable names.
                              used to load non-confirming files. **NotImplemented**
        :type grid_topology: mapping with keys of topology components and values are
                             variable names.

        Either a filename or grid and variable objects should be provided -- not both.
        """
        if ncfile is not None:
            if (grid is not None or variables is not None or grid_topology is not None):
                raise ValueError("You can create a Dataset from a file, or from raw data"
                                 "but not both.")
            self.nc_dataset = get_dataset(ncfile)
            self.filename = self.nc_dataset.filepath()
            self.grid = Grid.from_netCDF(filename=self.filename,
                                         dataset=self.nc_dataset,
                                         grid_topology=grid_topology)
            self.variables = self._load_variables(self.nc_dataset)
        else:  # no file passed in -- create from grid and variables
            self.filename = None
            self.grid = grid
            self.variables = variables

    def _load_variables(self, ds):
        """
        load up the variables in the nc file
        """
        variables = {}
        for k in ds.variables.keys():
            is_not_grid_attr = all([k not in str(v).split()
                                    for v in self.grid.grid_topology.values()])
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

    # This should be covered by Grid.from_netCDF
    # def load_from_topology_varnames(self, ncfile, topology):
    #     """
    #     Load a Gridded dataset by specifying the variable names used for the topology

    #     :param ncfile: a file to load the Dataset from.
    #     :type ncfile: filename of netcdf file or opendap url or open netCDF4 Dataset object
    #                  (could be other file types in the future)

    #     :param topology: variables that define the topology
    #     :type topology: dict of topology_role keys, and variable name values

    #     Docs about what is required for each grid type here.

    #     NOTE: the grid type will be inferred by what topology is provided.
    #     """

    #     raise NotImplementedError

    @property
    def var_names(self):
        """
        The names of the variables in the Dataset

        The variables can be acceced via the variable dict:

        temp_var = ds.variables['temp']

        or directly with indexing:

        temp_var = ds['temp']
        """
        return list(self.variables.keys())

    @property
    def bounds(self):
        """
        The bounding box of the grid
        """
        try:
            return self.grid.bounds
        except AttributeError:
            raise NotImplementedError("%s does not have a bounds property" % type(self.grid))


    def __getitem__(self, var_name):
        try:
            return self.variables[var_name]
        except KeyError:
            raise ValueError("There is no variable named: %s" % var_name)

    def save(self, filename, format='netcdf4'):
        """
        save the dataset to a file

        :param filename: full path to file to save to.

        :param format: format to save -- 'netcdf3' or 'netcdf4'
                       are the only options at this point.
        """
        raise NotImplementedError

    def get_variables_by_attribute(self, attr, value):
        """
        return the variables that have attributes that fit the defined input

        :param attr: the name of the attribute you want to match

        :param value: the value of the attribute you want to match

        fixme: make this a bit more flexible, more like the netCDF4 version
        """
        variables = []
        for var in self.variables.values:
            try:
                if variables.attributes[attr] == value:
                    variables.append(var)
            except KeyError:
                pass
        return variables

