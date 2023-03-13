#!/usr/bin/env python

"""
gridded module:

This module defines the gridded.Dataset --
The core class that encapsulates the gridded data model

"""



# py2/3 compatibility

from gridded.grids import Grid
from gridded.variable import Variable

from gridded.utilities import (get_dataset,
                               get_writable_dataset,
                               get_dataset_attrs,
                               )
from . import VALID_LOCATIONS

"""
The main gridded.Dataset code
"""


class Dataset():
    """
    An object that represent an entire complete dataset --
    a collection of Variables and the Grid that they are stored on.
    """

    def __init__(self,
                 ncfile=None,
                 grid=None,
                 variables=None,
                 grid_topology=None,
                 attributes=None):
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

        :param variables: a dict
        of dataset.Variable objects -- or anything that
                          presents the same API.

        :param grid_topology: mapping of grid topology components to netcdf variable names.
                              used to load non-confirming files.
        :type grid_topology: mapping with keys of topology components and values are
                             variable names.

        :param attributes: The global attributes of the dataset -- usually the global
                           attributes of a netcdf file.
        :type attributes: Mapping of attribute name to attributes themselves
                          (usually strings)

        Either a filename or grid and variable objects should be provided -- not both.

        If a filename is passed in, the attributes will be pulled from the file, and
        the input ones ignored.
        """
        if ncfile is not None:
            if (grid is not None or
                  variables is not None or
                  attributes is not None):
                raise ValueError("You can create a Dataset from a file, or from raw data"
                                 "but not both.")
            self.nc_dataset = get_dataset(ncfile)
            self.filename = self.nc_dataset.filepath()
            self.grid = Grid.from_netCDF(filename=self.filename,
                                         dataset=self.nc_dataset,
                                         grid_topology=grid_topology)
            self.variables = self._load_variables(self.nc_dataset)
            self.attributes = get_dataset_attrs(self.nc_dataset)
        else:  # no file passed in -- create from grid and variables
            self.filename = None
            self.grid = grid
            self.variables = {} if variables is None else variables
            self.attributes = {} if attributes is None else attributes

    def __getitem__(self, key):
        """
        shortcut to getting a variable object
        """
        return self.variables[key]

    def _load_variables(self, ds):
        """
        load up the variables in the nc file
        """
        variables = {}
        for k in ds.variables.keys():
            # find which netcdf variables are used to define the grid
            is_not_grid_attr = all([k not in str(v).split()
                                    for v in self.grid.grid_topology.values()])
            if is_not_grid_attr:
                ncvar = ds[k]
                # find the location of the variable
                # print("working with:", ncvar)
                try:
                    location = ncvar.location
                    if location not in VALID_LOCATIONS:
                        raise AttributeError("not a valid location name")
                except AttributeError:
                    # that didn't work, need to try to infer it
                    location = self.grid.infer_location(ncvar)
                if location is not None:
                    try:
                        ln = ds[k].long_name
                    except AttributeError:  # no long_name attribute
                        ln = ds[k].name # use the name attribute
                    # fixme: Variable.from_netCDF should really be able to figure out the location itself
                    #        maybe we need multiple Variable subclasses for different grid types?
                    #        CHB: yes, we really should do that!
                    variables[k] = Variable.from_netCDF(dataset=ds,
                                                        name=ln,
                                                        varname=k,
                                                        grid=self.grid,
                                                        location=location,
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

    def save(self, filename, format='netcdf4'):
        """
        save the dataset to a file

        :param filename: full path to file to save to.

        :param format: format to save -- 'netcdf3' or 'netcdf4'
                       are the only options at this point.
        """
        format_options = ('netcdf3', 'netcdf4')
        if format not in format_options:
            raise ValueError("format: {} not supported. Options are: {}".format(format, format_options))

        # create an ncdataset
        ncds = get_writable_dataset(filename)

        # Save the grid and variables
        self.grid.save(ncds, format='netcdf4', variables=self.variables)

        ncds.close()

    def get_variables_by_attribute(self, attr, value):
        """
        return the variables that have attributes that fit the defined input

        :param attr: the name of the attribute you want to match

        :param value: the value of the attribute you want to match

        fixme: make this a bit more flexible, more like the netCDF4 version
        """
        variables = []
        for var in self.variables.values():
            try:
                if var.attributes[attr] == value:
                    variables.append(var)
            except KeyError:
                pass
        return variables

    @property
    def info(self):
        """
        Information about the Dataset object
        """
        vars = [var.info for var in self.variables.values()]
        vars = "".join([" " * 8 + v for v in vars])
        vars = "\n".join([" " * 8 + line for line in vars.split("\n")])
        attrs = "\n".join(["        {}: {}".format(k, v) for k, v in self.attributes.items()])
        grid = "\n".join([" " * 8 + line for line in self.grid.info.split("\n")])
        msg = ("gridded.Dataset:\n"
               "    filename: {0.filename}\n"
               "    grid:\n{3}\n"
               "    variables: {1}\n"
               "    attributes:\n{2}".format(self,
                                             vars,
                                             attrs,
                                             grid
                                             ))
        return msg
