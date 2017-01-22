#!/usr/bin/env python

# py2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals

import copy

import numpy as np
import netCDF4 as nc4
from . import pysgrid
from . import pyugrid

"""
The main gridded.Dataset code
"""

import netCDF4


class Dataset():
    """
    An object that represent an entire complete dataset -- a collection of Variable,
    and the grid that they are stored on.
    """

    def __init__(self, ncfile=None, grid=None, variables=None):
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
            self.nc_dataset = _get_dataset(ncfile)
            self.filename = self.nc_dataset.filepath
            self.grid = None
            self.variables = {}
        else:  # no file passed in -- create from grid and variables
            self.filename = None
            self.grid = grid
            self.variables = variables


class Variable():
    """
    Variable object: represents a field of values associated with the grid.

    Abstractly, it is usually a scalar physical property such a temperature,
    salinity that varies over a the domain of the model.


    This more or less maps to a variable in a netcdf file, but does not have
    to come form a netcdf file, and this provides and abstraction where the
    user can access the value in world coordinates, interpolated from the grid.

    It holds a reference to its own grid object, and its data.
    """
    def __init__(self, ):
        pass

# # pulled from pyugrid
# class UVar(object):
#     """
#     A class to hold a variable associated with the UGrid. Data can be on the
#     nodes, edges, etc. -- "UGrid Variable"

#     It holds an array of the data, as well as the attributes associated
#     with that data  -- this is mapped to a netcdf variable with
#     attributes(attributes get stored in the netcdf file)
#     """

#     def __init__(self, name, location, data=None, attributes=None):
#         """
#         create a UVar object
#         :param name: the name of the variable (depth, u_velocity, etc.)
#         :type name: string

#         :param location: the type of grid element the data is associated with:
#                          'node', 'edge', or 'face'

#         :param data: The data itself
#         :type data: 1-d numpy array or array-like object ().
#                     If you have a list or tuple, it should be something that can be
#                     converted to a numpy array (list, etc.)
#         """
#         self.name = name

#         if location not in ['node', 'edge', 'face', 'boundary']:
#             raise ValueError("location must be one of: "
#                              "'node', 'edge', 'face', 'boundary'")

#         self.location = location

#         if data is None:
#             # Could be any data type, but we'll default to float
#             self._data = np.zeros((0,), dtype=np.float64)
#         else:
#             self._data = asarraylike(data)

#         # FixMe: we need a separate attribute dict -- we really do'nt want all this
#         #        getting mixed up with the python object attributes
#         self.attributes = {} if attributes is None else attributes
#         # if the data is a netcdf variable, pull the attributes from there
#         try:
#             for attr in data.ncattrs():
#                 self.attributes[attr] = data.getncattr(attr)
#         except AttributeError:  # must not be a netcdf variable
#             pass

#         self._cache = OrderedDict()

#     # def update_attrs(self, attrs):
#     #     """
#     #     update the attributes of the UVar object

#     #     :param attr: Dict containing attributes to be added to the object
#     #     """
#     #     for key, val in attrs.items():
#     #         setattr(self, key, val)

#     @property
#     def data(self):
#         return self._data

#     @data.setter
#     def data(self, data):
#         self._data = asarraylike(data)

#     @data.deleter
#     def data(self):
#         self._data = self._data = np.zeros((0,), dtype=np.float64)

#     @property
#     def shape(self):
#         return self.data.shape

#     @property
#     def max(self):
#         return np.max(self._data)

#     @property
#     def min(self):
#         return np.min(self._data)

#     @property
#     def dtype(self):
#         return self.data.dtype

#     @property
#     def ndim(self):
#         return self.data.ndim

#     def __getitem__(self, item):
#         """
#         Transfers responsibility to the data's __getitem__ if not cached
#         """
#         rv = None
#         if str(item) in self._cache:
#             rv = self._cache[str(item)]
#         else:
#             rv = self._data.__getitem__(item)
#             self._cache[str(item)] = rv
#             if len(self._cache) > 3:
#                 self._cache.popitem(last=False)
#         return rv

#     def __str__(self):
#         print("in __str__, data is:", self.data)
#         msg = ("UVar object: {0:s}, on the {1:s}s, and {2:d} data "
#                "points\nAttributes: {3}").format
#         return msg(self.name, self.location, len(self.data), self.attributes)

#     def __len__(self):
#         return len(self.data)



class PyGrid(object):
    _def_count = 0

    def __new__(cls, *args, **kwargs):
        '''
        If you construct a PyGrid object directly, you will always
        get one of the child types based on your input
        '''
        if cls is not PyGrid_U and cls is not PyGrid_S:
            if 'faces' in kwargs:
                cls = PyGrid_U
            else:
                cls = PyGrid_S
#         cls.obj_type = c.obj_type
        return super(type(cls), cls).__new__(cls, *args, **kwargs)

    def __init__(self,
                 filename=None,
                 *args,
                 **kwargs):
        '''
        Init common to all PyGrid types. This constructor will take all the kwargs of both
        pyugrid.UGrid and pysgrid.SGrid. See their documentation for details

        :param filename: Name of the file this grid was constructed from, if available.
        '''
        super(PyGrid, self).__init__(**kwargs)
        if 'name' in kwargs:
            self.name = kwargs['name']
        else:
            self.name = self.__class__.__name__ + '_' + str(type(self)._def_count)
        self.obj_type = str(type(self).__bases__[0])
        self.filename = filename
        type(self)._def_count += 1

    @classmethod
    def load_grid(cls, filename, topology_var):
        '''
        Redirect to grid-specific loading routine.
        '''
        if hasattr(topology_var, 'face_node_connectivity') or isinstance(topology_var, dict) and 'faces' in topology_var.keys():
            cls = PyGrid_U
            return cls.from_ncfile(filename)
        else:
            cls = PyGrid_S
            return cls.load_grid(filename)
        pass

    @classmethod
    def from_netCDF(cls, filename=None, dataset=None, grid_type=None, grid_topology=None, *args, **kwargs):
        '''
        :param filename: File containing a grid
        :param dataset: Takes precedence over filename, if provided.
        :param grid_type: Must be provided if Dataset does not have a 'grid_type' attribute, or valid topology variable
        :param grid_topology: A dictionary mapping of grid attribute to variable name. Takes precendence over discovered attributes
        :param **kwargs: All kwargs to SGrid or UGrid are valid, and take precedence over all.
        :returns: Instance of PyGrid_U, PyGrid_S, or PyGrid_R
        '''
        gf = dataset if filename is None else _get_dataset(filename, dataset)
        if gf is None:
            raise ValueError('No filename or dataset provided')

        cls = PyGrid._get_grid_type(gf, grid_topology, grid_type)
        init_args, gf_vars = cls._find_required_grid_attrs(filename,
                                                           dataset=dataset,
                                                           grid_topology=grid_topology)
        return cls(**init_args)

    @classmethod
    def _find_required_grid_attrs(cls, filename, dataset=None, grid_topology=None,):
        '''
        This function is the top level 'search for attributes' function. If there are any
        common attributes to all potential grid types, they will be sought here.

        This function returns a dict, which maps an attribute name to a netCDF4
        Variable or numpy array object extracted from the dataset. When called from
        PyGrid_U or PyGrid_S, this function should provide all the kwargs needed to
        create a valid instance.
        '''
        gf_vars = dataset.variables if dataset is not None else _get_dataset(filename).variables
        init_args = {}
        init_args['filename'] = filename
        node_attrs = ['node_lon', 'node_lat']
        node_coord_names = [['node_lon', 'node_lat'], ['lon', 'lat'], ['lon_psi', 'lat_psi']]
        composite_node_names = ['nodes', 'node']
        if grid_topology is None:
            for n1, n2 in node_coord_names:
                if n1 in gf_vars and n2 in gf_vars:
                    init_args[node_attrs[0]] = gf_vars[n1][:]
                    init_args[node_attrs[1]] = gf_vars[n2][:]
                    break
            if node_attrs[0] not in init_args:
                for n in composite_node_names:
                    if n in gf_vars:
                        v = gf_vars[n][:].reshape(-1, 2)
                        init_args[node_attrs[0]] = v[:, 0]
                        init_args[node_attrs[1]] = v[:, 1]
                        break
            if node_attrs[0] not in init_args:
                raise ValueError('Unable to find node coordinates.')
        else:
            for n, v in grid_topology.items():
                if n in node_attrs:
                    init_args[n] = gf_vars[v][:]
                if n in composite_node_names:
                    v = gf_vars[n][:].reshape(-1, 2)
                    init_args[node_attrs[0]] = v[:, 0]
                    init_args[node_attrs[1]] = v[:, 1]
        return init_args, gf_vars

    @classmethod
    def new_from_dict(cls, dict_):
        dict_.pop('json_')
        filename = dict_['filename']
        rv = cls.from_netCDF(filename)
        rv.__class__._restore_attr_from_save(rv, dict_)
        rv._id = dict_.pop('id') if 'id' in dict_ else rv.id
        rv.__class__._def_count -= 1
        return rv

    @staticmethod
    def _get_grid_type(dataset, grid_topology=None, grid_type=None):
        sgrid_names = ['sgrid', 'pygrid_s', 'staggered', 'curvilinear', 'roms']
        ugrid_names = ['ugrid', 'pygrid_u', 'triangular', 'unstructured']
        if grid_type is not None:
            if grid_type.lower() in sgrid_names:
                return PyGrid_S
            elif grid_type.lower() in ugrid_names:
                return PyGrid_U
            else:
                raise ValueError('Specified grid_type not recognized/supported')
        if grid_topology is not None:
            if 'faces' in grid_topology.keys() or grid_topology.get('grid_type', 'notype').lower() in ugrid_names:
                return PyGrid_U
            else:
                return PyGrid_S
        else:
            # no topology, so search dataset for grid_type variable
            if hasattr(dataset, 'grid_type') and dataset.grid_type in sgrid_names + ugrid_names:
                if dataset.grid_type.lower() in ugrid_names:
                    return PyGrid_U
                else:
                    return PyGrid_S
            else:
                # no grid type explicitly specified. is a topology variable present?
                topology = PyGrid._find_topology_var(None, dataset=dataset)
                if topology is not None:
                    if hasattr(topology, 'node_coordinates') and not hasattr(topology, 'node_dimensions'):
                        return PyGrid_U
                    else:
                        return PyGrid_S
                else:
                    # no topology variable either, so generate and try again.
                    # if no defaults are found, _gen_topology will raise an error
                    try:
                        u_init_args, u_gf_vars = PyGrid_U._find_required_grid_attrs(None, dataset)
                        return PyGrid_U
                    except ValueError:
                        s_init_args, s_gf_vars = PyGrid_S._find_required_grid_attrs(None, dataset)
                        return PyGrid_S

    @staticmethod
    def _find_topology_var(filename,
                           dataset=None):
        gf = _get_dataset(filename, dataset)
        gts = []
        for v in gf.variables:
            if hasattr(v, 'cf_role') and 'topology' in v.cf_role:
                gts.append(v)
#         gts = gf.get_variables_by_attributes(cf_role=lambda t: t is not None and 'topology' in t)
        if len(gts) != 0:
            return gts[0]
        else:
            return None

    @property
    def shape(self):
        return self.node_lon.shape

    def __eq__(self, o):
        if self is o:
            return True
        for n in ('nodes', 'faces'):
            if hasattr(self, n) and hasattr(o, n) and getattr(self, n) is not None and getattr(o, n) is not None:
                s = getattr(self, n)
                s2 = getattr(o, n)
                if s.shape != s2.shape or np.any(s != s2):
                    return False
        return True

    def _write_grid_to_file(self, pth):
        self.save_as_netcdf(pth)

    def save(self, saveloc, references=None, name=None):
        '''
        INCOMPLETE
        Write Wind timeseries to file or to zip,
        then call save method using super
        '''
#         name = self.name
#         saveloc = os.path.splitext(name)[0] + '_grid.GRD'

        if zipfile.is_zipfile(saveloc):
            if self.filename is None:
                self._write_grid_to_file(saveloc)
                self._write_grid_to_zip(saveloc, saveloc)
                self.filename = saveloc
#             else:
#                 self._write_grid_to_zip(saveloc, self.filename)
        else:
            if self.filename is None:
                self._write_grid_to_file(saveloc)
                self.filename = saveloc
        return super(PyGrid, self).save(saveloc, references, name)


class PyGrid_U(PyGrid, pyugrid.UGrid):

    @classmethod
    def _find_required_grid_attrs(cls, filename, dataset=None, grid_topology=None):

        # Get superset attributes
        init_args, gf_vars = super(PyGrid_U, cls)._find_required_grid_attrs(filename=filename,
                                                                            dataset=dataset,
                                                                            grid_topology=grid_topology)

        face_attrs = ['faces']
        if grid_topology is not None:
            face_var_names = [grid_topology.get(n) for n in face_attrs]
        else:
            face_var_names = ['faces', 'tris', 'nv', 'ele']

        for n in face_var_names:
            if n in gf_vars:
                init_args[face_attrs[0]] = gf_vars[n][:]
                break
        if face_attrs[0] in init_args:
            if init_args[face_attrs[0]].shape[0] == 3:
                init_args[face_attrs[0]] = np.ascontiguousarray(np.array(init_args[face_attrs[0]]).T - 1)
            return init_args, gf_vars
        else:
            raise ValueError('Unable to find faces variable')


class PyGrid_S(PyGrid, pysgrid.SGrid):

    @classmethod
    def _find_required_grid_attrs(cls, filename, dataset=None, grid_topology=None):

        # THESE ARE ACTUALLY ALL OPTIONAL. This should be migrated when optional attributes are dealt with
        # Get superset attributes
        init_args, gf_vars = super(PyGrid_S, cls)._find_required_grid_attrs(filename,
                                                                            dataset=dataset,
                                                                            grid_topology=grid_topology)

        center_attrs = ['center_lon', 'center_lat']
        edge1_attrs = ['edge1_lon', 'edge1_lat']
        edge2_attrs = ['edge2_lon', 'edge2_lat']

        center_coord_names = [['center_lon', 'center_lat'], ['lon_rho', 'lat_rho']]
        edge1_coord_names = [['edge1_lon', 'edge1_lat'], ['lon_u', 'lat_u']]
        edge2_coord_names = [['edge2_lon', 'edge2_lat'], ['lon_v', 'lat_v']]

        if grid_topology is None:
            for attr, names in (zip((center_attrs, edge1_attrs, edge2_attrs),
                                    (center_coord_names, edge1_coord_names, edge2_coord_names))):
                for n1, n2 in names:
                    if n1 in gf_vars and n2 in gf_vars:
                        init_args[attr[0]] = gf_vars[n1][:]
                        init_args[attr[1]] = gf_vars[n2][:]
                        break
        else:
            for n, v in grid_topology.items():
                if n in center_attrs + edge1_attrs + edge2_attrs and v in gf_vars:
                    init_args[n] = gf_vars[v][:]
        return init_args, gf_vars


def _get_dataset(ncfile, dataset=None):
    """
    Utility to create a netCDF4 Dataset from a filename, list of filenames,
    or just pass it through if it's already a netCDF4.Dataset

    if dataset is not None, it should be a valid netCDF4 Dataset object,
    and it will simiply be returned
    """
    if dataset is not None:
        return dataset
    if isinstance(ncfile, nc4.Dataset):
        return ncfile
    elif isinstance(ncfile, basestring):
        return nc4.Dataset(ncfile)
    else:
        return nc4.MFDataset(ncfile)


Grid = PyGrid
