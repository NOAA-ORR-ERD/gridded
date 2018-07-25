from __future__ import absolute_import, division, print_function, unicode_literals

import netCDF4 as nc4
import numpy as np
import collections
from collections import OrderedDict
from gridded.utilities import get_dataset, _reorganize_spatial_data,\
    _align_results_to_spatial_data
from gridded.grids import Grid, Grid_U, Grid_S, Grid_R
from gridded.depth import Depth
from gridded.time import Time

import hashlib
from functools import wraps
import pdb


class Variable(object):
    """
    Variable object: represents a field of values associated with the grid.

    Abstractly, it is usually a scalar physical property such a temperature,
    salinity that varies over a the domain of the model.


    This more or less maps to a variable in a netcdf file, but does not have
    to come form a netcdf file, and this provides and abstraction where the
    user can access the value in world coordinates, interpolated from the grid.

    It holds a reference to its own grid object, and its data.
    """
    default_names = []
    cf_names = []
    _def_count = 0

    _default_component_types = {'time': Time,
                                'grid': Grid,
                                'depth': Depth}

    def __init__(self,
                 name=None,
                 units=None,
                 time=None,
                 data=None,
                 grid=None,
                 depth=None,
                 data_file=None,
                 grid_file=None,
                 dataset=None,
                 varname=None,
                 fill_value=0,
                 attributes=None,
                 **kwargs):
        '''
        This class represents a phenomenon using gridded data

        :param name: Name
        :type name: string

        :param units: Units
        :type units: string

        :param time: Time axis of the data
        :type time: list of datetime.datetime, netCDF4 Variable, or Time object

        :param data: Underlying data source
        :type data: array-like object such as netCDF4.Variable or numpy.ndarray

        :param grid: Grid that the data corresponds with
        :type grid: GRid object (pysgrid or pyugrid or )

        :param data_file: Name of data source file
        :type data_file: string

        :param grid_file: Name of grid source file
        :type grid_file: string

        :param varname: Name of the variable in the data source file
        :type varname: string

        :param fill_value: the fill value used for undefined data

        :param attributes: attributes associated with the VAriable
                           (analogous to netcdf variable attributes)
        :type attributes: dict of key:value pairs
        '''

#         if any([grid is None, data is None]):
#             raise ValueError("Grid and Data must be defined")
#         if not hasattr(data, 'shape'):
#             if grid.infer_location is None:
#                 raise ValueError('Data must be able to fit to the grid')

        self.grid = grid
        self.depth = depth
        self.name = self._units = self._time = self._data = None

        self.name = name
        self.units = units
        self.data = data
        self.time = time if time is not None else self._default_component_types['time'].constant_time()
        self.data_file = data_file
        self.grid_file = grid_file
        self.varname = varname
        self._result_memo = collections.OrderedDict()
        self.fill_value = fill_value

        self.attributes = {} if attributes is None else attributes
        # if the data is a netcdf variable, pull the attributes from there
        try:
            for attr in self.data.ncattrs():
                self.attributes[attr] = data.getncattr(attr)
        except AttributeError:  # must not be a netcdf variable
            pass                # so just use what was passed in.

#         for k in kwargs:
#             setattr(self, k, kwargs[k])


    @classmethod
    def from_netCDF(cls,
                    filename=None,
                    varname=None,
                    grid_topology=None,
                    name=None,
                    units=None,
                    time=None,
                    time_origin=None,
                    grid=None,
                    depth=None,
                    dataset=None,
                    data_file=None,
                    grid_file=None,
                    load_all=False,
                    fill_value=0,
                    **kwargs
                    ):
        '''
        Allows one-function creation of a Variable from a file.

        :param filename: Default data source. Parameters below take precedence
        :param varname: Name of the variable in the data source file
        :param grid_topology: Description of the relationship between grid attributes and variable names.
        :param name: Name of property
        :param units: Units
        :param time: Time axis of the data
        :param data: Underlying data source
        :param grid: Grid that the data corresponds with
        :param depth: Depth axis object
        :param dataset: Instance of open Dataset
        :param data_file: Name of data source file
        :param grid_file: Name of grid source file
        :type filename: string
        :type varname: string
        :type grid_topology: {string : string, ...}
        :type name: string
        :type units: string
        :type time: [] of datetime.datetime, netCDF4 Variable, or Time object
        :type data: netCDF4.Variable or numpy.array
        :type grid: pysgrid or pyugrid
        :type depth: Depth, S_Depth or L_Depth
        :type dataset: netCDF4.Dataset
        :type data_file: string
        :type grid_file: string
        '''
        Grid = cls._default_component_types['grid']
        Time = cls._default_component_types['time']
        Depth = cls._default_component_types['depth']
        if filename is not None:
            data_file = filename
            grid_file = filename

        ds = None
        dg = None
        if dataset is None:
            if grid_file == data_file:
                ds = dg = get_dataset(grid_file)
            else:
                ds = get_dataset(data_file)
                dg = get_dataset(grid_file)
        else:
            if grid_file is not None:
                dg = get_dataset(grid_file)
            else:
                dg = dataset
            ds = dataset

        if grid is None:
            grid = Grid.from_netCDF(grid_file,
                                      dataset=dg,
                                      grid_topology=grid_topology)
        if varname is None:
            varname = cls._gen_varname(data_file,
                                       dataset=ds)
            if varname is None:
                raise NameError('Default current names are not in the data file, must supply variable name')
        data = ds[varname]
        if name is None:
            name = cls.__name__ + str(cls._def_count)
            cls._def_count += 1
        if units is None:
            try:
                units = data.units
            except AttributeError:
                units = None
        if time is None:
            time = Time.from_netCDF(filename=data_file,
                                    dataset=ds,
                                    datavar=data)
            if time_origin is not None:
                time = Time(data=time.data, filename=time.filename, varname=time.varname, origin=time_origin)
        if depth is None:
            if (isinstance(grid, (Grid_S, Grid_R)) and len(data.shape) == 4 or
                    isinstance(grid, Grid_U) and len(data.shape) == 3):
                depth = Depth.from_netCDF(grid_file,
                                          dataset=dg,
                                          )
#             if len(data.shape) == 4 or (len(data.shape) == 3 and time is None):
#                 from gnome.environment.environment_objects import S_Depth
#                 depth = S_Depth.from_netCDF(grid=grid,
#                                             depth=1,
#                                             data_file=data_file,
#                                             grid_file=grid_file,
#                                             **kwargs)
        if load_all:
            data = data[:]
        return cls(name=name,
                   units=units,
                   time=time,
                   data=data,
                   grid=grid,
                   depth=depth,
                   grid_file=grid_file,
                   data_file=data_file,
                   fill_value=fill_value,
                   varname=varname,
                   **kwargs)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return ('{0.__class__.__module__}.{0.__class__.__name__}('
                'name="{0.name}", '
                'time="{0.time}", '
                'units="{0.units}", '
                'data="{0.data}", '
                ')').format(self)

    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, t):
        Time = self.__class__._default_component_types['time']
        if t is None:
            self._time = None
            return
        if self.data is not None and len(t) != self.data.shape[0] and len(t) > 1:
            raise ValueError("Data/time interval mismatch")
        if isinstance(t, Time):
            self._time = t
        elif isinstance(t, collections.Iterable) or isinstance(t, nc4.Variable):
            self._time = Time(t)
        else:
            raise ValueError("Time must be set with an iterable container or netCDF variable")

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, d):
        if self.time is not None and len(d) != len(self.time):
            raise ValueError("Data/time interval mismatch")
        if self.grid is not None and self.grid.infer_location(d) is None:
            raise ValueError("Data/grid shape mismatch. Data shape is {0}, Grid shape is {1}".format(d.shape, self.grid.node_lon.shape))
        self._data = d

    @property
    def units(self):
        '''
        Units of underlying data

        :rtype: string
        '''
        return self._units

    @units.setter
    def units(self, unit):
#         if unit is not None:
#             if not unit_conversion.is_supported(unit):
#                 raise ValueError('Units of {0} are not supported'.format(unit))
        self._units = unit

    @property
    def grid_shape(self):
        if hasattr(self.grid, 'shape'):
            return self.grid.shape
        else:
            return self.grid.node_lon.shape

    @property
    def data_shape(self):
        return self.data.shape

    @property
    def is_data_on_nodes(self):
        return self.grid.infer_location(self._data) == 'node'

    def _get_hash(self, points, time):
        """
        Returns a SHA1 hash of the array of points passed in
        """
        return (hashlib.sha1(points.tobytes()).hexdigest(),
                hashlib.sha1(str(time).encode('utf-8')).hexdigest())

    def _memoize_result(self, points, time, result, D, _copy=False, _hash=None):
        if _copy:
            result = result.copy()
        result.setflags(write=False)
        if _hash is None:
            _hash = self._get_hash(points, time)
        if D is not None and len(D) > 4:
            D.popitem(last=False)
        D[_hash] = result
        D[_hash].setflags(write=False)

    def _get_memoed(self, points, time, D, _copy=False, _hash=None):
        if _hash is None:
            _hash = self._get_hash(points, time)
        if (D is not None and _hash in D):
            return D[_hash].copy() if _copy else D[_hash]
        else:
            return None

    def center_values(self, time, units=None, extrapolate=False):
        # NOT COMPLETE
        if not extrapolate:
            self.time.valid_time(time)
        if len(self.time) == 1:
            if len(self.data.shape) == 2:
                if isinstance(self.grid, Grid_S):
                    # curv grid
                    value = self.data[0:1:-2, 1:-2]
                else:
                    value = self.data
        else:
            centers = self.grid.get_center_points()
            value = self.at(centers, time, units)
        return value

    @property
    def dimension_ordering(self):
        '''
        Returns a list that describes the dimensions of the property's data. If a dimension_ordering is assigned,
        it will continue to use that. If no dimension_ordering is set, then a default ordering will be generated
        based on the object properties and data shape.

        For example, if the data has 4 dimensions and is represented by a Grid_S (structured grid), and the
        Variable has a depth and time assigned, then the assumed ordering is ['time','depth','lon','lat']

        If the data has 3 dimensions, self.grid is a Grid_S, and self.time is None, then the ordering is
        ['depth','lon','lat']
        If the data has 3 dimensions, self.grid is a Grid_U, the ordering is ['time','depth','ele']
        '''
        if not hasattr(self, '_order'):
            self._order = None
        if self._order is not None:
            return self._order
        else:
            if isinstance(self.grid, (Grid_S, Grid_R)):
                order = ['time', 'depth', 'lon', 'lat']
            else:
                order = ['time', 'depth', 'ele']
            ndim = len(self.data.shape)
            diff = len(order) - ndim
            if diff == 0:
                return order
            elif diff == 1:
                if self.time is not None:
                    del order[1]
                elif self.depth is not None:
                    del order[0]
                else:
                    raise ValueError('Generated ordering too short to fit data. Time or depth must not be None')
            elif diff == 2:
                order = order[2:]
            else:
                raise ValueError('Too many/too few dimensions ndim={0}'.format(ndim))
            return order

    @dimension_ordering.setter
    def dimension_ordering(self, order):
        self._order = order

#     @profile
    def at(self, points, time, units=None, extrapolate=False, _hash=None, _mem=True, _auto_align=True, unmask=False, **kwargs):
        '''
        Find the value of the property at positions P at time T

        :param points: Coordinates to be queried (P)
        :param time: The time at which to query these points (T)
        :param units: units the values will be returned in (or converted to)
        :param extrapolate: if True, extrapolation will be supported
        :type points: Nx2 array of double
        :type time: datetime.datetime object
        :type depth: integer
        :type units: string such as ('mem/s', 'knots', etc)
        :type extrapolate: boolean (True or False)
        :return: returns a Nx1 array of interpolated values
        :rtype: double
        '''
        pts = _reorganize_spatial_data(points)

        if _hash is None:
            _hash = self._get_hash(pts, time)

        if _mem:
            res = self._get_memoed(pts, time, self._result_memo, _hash=_hash)
            if res is not None:
                return res

        order = self.dimension_ordering
        if order[0] == 'time':
            value = self._time_interp(pts, time, extrapolate, _mem=_mem, _hash=_hash, **kwargs)
        elif order[0] == 'depth':
            value = self._depth_interp(pts, time, extrapolate, _mem=_mem, _hash=_hash, **kwargs)
        else:
            value = self._xy_interp(pts, time, extrapolate, _mem=_mem, _hash=_hash, **kwargs)

        if _auto_align == True:
            value = _align_results_to_spatial_data(value.copy(), points)

        if isinstance(value, np.ma.MaskedArray):
            np.ma.set_fill_value(value, self.fill_value)
        if unmask:
            value = np.ma.filled(value)

        if _mem:
            self._memoize_result(pts, time, value, self._result_memo, _hash=_hash)

        return value

    def _xy_interp(self, points, time, extrapolate, slices=(), **kwargs):
        '''
        Uses the py(s/u)grid interpolation to determine the values at the points, and returns it
        :param points: Coordinates to be queried (3D)
        :param time: Time of the query
        :param extrapolate: Turns extrapolation on or off
        :param slices: describes how the data needs to be sliced to reach the appropriate dimension
        :type points: Nx3 array of double
        :type time: datetime.datetime object
        :type extrapolate: boolean
        :type slices: tuple of integers or slice objects
        '''
        _hash = kwargs['_hash'] if '_hash' in kwargs else None
        units = kwargs['units'] if 'units' in kwargs else None
        value = self.grid.interpolate_var_to_points(points[:, 0:2], self.data, _hash=self._get_hash(points[:,0:2], time), slices=slices, _memo=True)
        return value

    def _time_interp(self, points, time, extrapolate, slices=(), **kwargs):
        '''
        Uses the Time object to interpolate the result of the next level of interpolation, as specified
        by the dimension_ordering attribute.
        :param points: Coordinates to be queried (3D)
        :param time: Time of the query
        :param extrapolate: Turns extrapolation on or off
        :param slices: describes how the data needs to be sliced to reach the appropriate dimension
        :type points: Nx3 array of double
        :type time: datetime.datetime object
        :type extrapolate: boolean
        :type slices: tuple of integers or slice objects
        '''
        order = self.dimension_ordering
        idx = order.index('time')
        if order[idx + 1] != 'depth':
            val_func = self._xy_interp
        else:
            val_func = self._depth_interp

        if time == self.time.min_time or (extrapolate and time < self.time.min_time):
            # min or before
            return val_func(points, time, extrapolate, slices=(0,), ** kwargs)
        elif time == self.time.max_time or (extrapolate and time > self.time.max_time):
            return val_func(points, time, extrapolate, slices=(-1,), **kwargs)
        else:
            ind = self.time.index_of(time)
            s1 = slices + (ind,)
            s0 = slices + (ind - 1,)
            v0 = val_func(points, time, extrapolate, slices=s0, **kwargs)
            v1 = val_func(points, time, extrapolate, slices=s1, **kwargs)
            alphas = self.time.interp_alpha(time)
            value = v0 + (v1 - v0) * alphas
            return value

    def _depth_interp(self, points, time, extrapolate, slices=(), **kwargs):
        '''
        Uses the Depth object to interpolate the result of the next level of interpolation, as specified
        by the dimension_ordering attribute.
        :param points: Coordinates to be queried (3D)
        :param time: Time of the query
        :param extrapolate: Turns extrapolation on or off
        :param slices: describes how the data needs to be sliced to reach the appropriate dimension
        :type points: Nx3 array of double
        :type time: datetime.datetime object
        :type extrapolate: boolean
        :type slices: tuple of integers or slice objects
        '''
        order = self.dimension_ordering
        idx = order.index('depth')
        if order[idx + 1] != 'time':
            val_func = self._xy_interp
        else:
            val_func = self._time_interp
        indices, alphas = self.depth.interpolation_alphas(points, time, self.data.shape[1:], kwargs.get('_hash', None))
        if indices is None and alphas is None:
            # all particles are on surface
            return val_func(points, time, extrapolate, slices=slices + (self.depth.surface_index,), **kwargs)
        else:
            min_idx = indices[indices != -1].min() - 1
            max_idx = indices.max()
            values = np.zeros(len(points), dtype=np.float64)
            v0 = val_func(points, time, extrapolate, slices=slices + (min_idx - 1,), **kwargs)
            for idx in range(min_idx + 1, max_idx + 1):
                v1 = val_func(points, time, extrapolate, slices=slices + (idx,), **kwargs)
                pos_idxs = np.where(indices == idx)[0]
                sub_vals = v0 + (v1 - v0) * alphas
                if len(pos_idxs) > 0:
                    values.put(pos_idxs, sub_vals.take(pos_idxs))
                v0 = v1
            if extrapolate:
                underground = (indices == self.depth.bottom_index)
                values[underground] = val_func(points, time, extrapolate, slices=slices + (self.depth.bottom_index,), **kwargs)
            else:
                underground = (indices == self.depth.bottom_index)
                values[underground] = self.fill_value
            return values

    def transect(self, times, depths, points):
        output_shape = (len(times), len(depths), len(points))
        outarr = np.array(shape=output_shape)
        for t in range(0,len(times)):
            for d in range(0, len(depths)):
                pts = np.array(shape=(len(points),3))
                pts[:,0:2] = points
                pts[:,2] = depths[d]
                layer = d

    @classmethod
    def _gen_varname(cls,
                     filename=None,
                     dataset=None,
                     names_list=None,
                     std_names_list=None):
        """
        Function to find the default variable names if they are not provided.

        :param filename: Name of file that will be searched for variables
        :param dataset: Existing instance of a netCDF4.Dataset
        :type filename: string
        :type dataset: netCDF.Dataset
        :return: List of default variable names, or None if none are found
        """
        df = None
        if dataset is not None:
            df = dataset
        else:
            df = get_dataset(filename)
        if names_list is None:
            names_list = cls.default_names
        if std_names_list is None:
            std_names_list = cls.cf_names
        for n in names_list:
            if n in df.variables.keys():
                return n
        for n in std_names_list:
            for var in df.variables.values():
                if (hasattr(var, 'standard_name') and var.standard_name == n or
                        hasattr(var, 'long_name') and var.long_name == n):
                    return var.name
        raise ValueError("Default names not found.")


class VectorVariable(object):

    default_names = {}
    cf_names = {}
    comp_order=[]

    _def_count = 0

    ''''
    These are the classes which are used when internal components are created
    by default, such as automatically from a file or other python data structure.
    Subclasses of this type may override this to use different classes for it's
    components
    '''
    _default_component_types = {'time': Time,
                                'grid': Grid,
                                'depth': Depth,
                                'variable': Variable}
    def __init__(self,
                 name=None,
                 units=None,
                 time=None,
                 variables=None,
                 grid=None,
                 depth=None,
                 grid_file=None,
                 data_file=None,
                 dataset=None,
                 varnames=None,
                 **kwargs):

        self.name = self._units = self._time = self._variables = None

        self.name = name

        if all([isinstance(v, Variable) for v in variables]):
            if time is not None and not isinstance(time, Time):
                time = Time(time)
            units = variables[0].units if units is None else units
            time = variables[0].time if time is None else time
        if units is None:
            units = variables[0].units
        self._units = units
        if variables is None or len(variables) < 2:
            raise ValueError('Variables must be an array-like of 2 or more Variable objects')
        self.variables = variables
        self._time = time
        unused_args = kwargs.keys() if kwargs is not None else None
        if len(unused_args) > 0:
            kwargs = {}
        if isinstance(self.variables[0], Variable):
            self.grid = self.variables[0].grid if grid is None else grid
            self.depth = self.variables[0].depth if depth is None else depth
            self.grid_file = self.variables[0].grid_file if grid_file is None else grid_file
            self.data_file = self.variables[0].data_file if data_file is None else data_file

        self._result_memo = OrderedDict()
        for i, comp in enumerate(self.__class__.comp_order):
            setattr(self, comp, self.variables[i])

    @classmethod
    def from_netCDF(cls,
                    filename=None,
                    varnames=None,
                    grid_topology=None,
                    name=None,
                    units=None,
                    time=None,
                    time_origin=None,
                    grid=None,
                    depth=None,
                    data_file=None,
                    grid_file=None,
                    dataset=None,
                    load_all=False,
                    variables=None,
                    **kwargs
                    ):
        '''
        Allows one-function creation of a VectorVariable from a file.

        :param filename: Default data source. Parameters below take precedence
        :param varnames: Names of the variables in the data source file
        :param grid_topology: Description of the relationship between grid attributes and variable names.
        :param name: Name of property
        :param units: Units
        :param time: Time axis of the data
        :param data: Underlying data source
        :param grid: Grid that the data corresponds with
        :param dataset: Instance of open Dataset
        :param data_file: Name of data source file
        :param grid_file: Name of grid source file
        :type filename: string
        :type varnames: [] of string
        :type grid_topology: {string : string, ...}
        :type name: string
        :type units: string
        :type time: [] of datetime.datetime, netCDF4 Variable, or Time object
        :type data: netCDF4.Variable or numpy.array
        :type grid: pysgrid or pyugrid
        :type dataset: netCDF4.Dataset
        :type data_file: string
        :type grid_file: string
        '''
        Grid = cls._default_component_types['grid']
        Time = cls._default_component_types['time']
        Variable = cls._default_component_types['variable']
        Depth = cls._default_component_types['depth']
        if filename is not None:
            data_file = filename
            grid_file = filename

        ds = None
        dg = None
        if dataset is None:
            if grid_file == data_file:
                ds = dg = get_dataset(grid_file)
            else:
                ds = get_dataset(data_file)
                dg = get_dataset(grid_file)
        else:
            if grid_file is not None:
                dg = get_dataset(grid_file)
            else:
                dg = dataset
            ds = dataset

        if grid is None:
            grid = Grid.from_netCDF(grid_file,
                                      dataset=dg,
                                      grid_topology=grid_topology)
        if varnames is None:
            varnames = cls._gen_varnames(data_file,
                                         dataset=ds)
            if all([v is None for v in varnames]):
                raise ValueError('No compatible variable names found!')
        if name is None:
            name = cls.__name__ + str(cls._def_count)
            cls._def_count += 1
        data = ds[varnames[0]]
        if time is None:
            time = Time.from_netCDF(filename=data_file,
                                    dataset=ds,
                                    datavar=data)
            if time_origin is not None:
                time = Time(data=time.data, filename=data_file, varname=time.varname, origin=time_origin)
        if depth is None:
            if (isinstance(grid, (Grid_S, Grid_R)) and len(data.shape) == 4 or
                    isinstance(grid, Grid_U) and len(data.shape) == 3):
                depth = Depth.from_netCDF(grid_file,
                                          dataset=dg,
                                          )

#         if depth is None:
#             if (isinstance(grid, Grid_S) and len(data.shape) == 4 or
#                         (len(data.shape) == 3 and time is None) or
#                     (isinstance(grid, Grid_U) and len(data.shape) == 3 or
#                         (len(data.shape) == 2 and time is None))):
#                 from gnome.environment.environment_objects import S_Depth
#                 depth = S_Depth.from_netCDF(grid=grid,
#                                             depth=1,
#                                             data_file=data_file,
#                                             grid_file=grid_file,
#                                             **kwargs)
        if variables is None:
            variables = []
            for vn in varnames:
                if vn is not None:
                    variables.append(Variable.from_netCDF(filename=filename,
                                                          varname=vn,
                                                          grid_topology=grid_topology,
                                                          units=units,
                                                          time=time,
                                                          grid=grid,
                                                           depth=depth,
                                                          data_file=data_file,
                                                          grid_file=grid_file,
                                                          dataset=ds,
                                                          load_all=load_all,
                                                          **kwargs))
        if units is None:
            units = [v.units for v in variables]
            if all(u == units[0] for u in units):
                units = units[0]
        return cls(name=name,
                   filename=filename,
                   varnames=varnames,
                   grid_topology=grid_topology,
                   units=units,
                   time=time,
                   grid=grid,
                   depth=depth,
                   variables=variables,
                   data_file=data_file,
                   grid_file=grid_file,
                   dataset=ds,
                   load_all=load_all,
                   **kwargs)

    @classmethod
    def _gen_varnames(cls,
                      filename=None,
                      dataset=None,
                      names_dict=None,
                      std_names_dict=None):
        """
        Function to find the default variable names if they are not provided.

        :param filename: Name of file that will be searched for variables
        :param dataset: Existing instance of a netCDF4.Dataset
        :type filename: string
        :type dataset: netCDF.Dataset
        :return: dict of component to name mapping (eg {'u': 'water_u', 'v': 'water_v', etc})
        """
        df = None
        if dataset is not None:
            df = dataset
        else:
            df = get_dataset(filename)
        if names_dict is None:
            names_dict = cls.default_names
        if std_names_dict is None:
            std_names_dict = cls.cf_names
        rd = {}
        for k in cls.comp_order:
            v = names_dict[k] if k in names_dict else []
            for n in v:
                if n in df.variables.keys():
                    rd[k] = n
                    continue
            if k not in rd.keys():
                rd[k] = None
        for k in cls.comp_order:
            v = std_names_dict[k] if k in std_names_dict else []
            if rd[k] is None:
                for n in v:
                    for var in df.variables.values():
                        if (hasattr(var, 'standard_name') and var.standard_name == n or
                                hasattr(var, 'long_name') and var.long_name == n):
                            rd[k] = var.name
                            break
        return collections.namedtuple('varnames', cls.comp_order)(**rd)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return ('{0.__class__.__module__}.{0.__class__.__name__}('
                'name="{0.name}", '
                'time="{0.time}", '
                'units="{0.units}", '
                'variables="{0.variables}", '
                'grid="{0.grid}", '
                ')').format(self)

    @property
    def is_data_on_nodes(self):
        return self.grid.infer_location(self.variables[0].data) == 'node'

    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, t):
        Time = self.__class__._default_component_types['time']
        if self.variables is not None:
            for v in self.variables:
                try:
                    v.time = t
                except ValueError as e:
                    raise ValueError('''Time was not compatible with variables.
                    Set variables attribute to None to allow changing other attributes
                    Original error: {0}'''.format(str(e)))
        if isinstance(t, Time):
            self._time = t
        elif isinstance(t, collections.Iterable) or isinstance(t, nc4.Variable):
            self._time = Time(t)
        else:
            raise ValueError("Time must be set with an iterable container or netCDF variable")

    @property
    def units(self):
        '''
        Units of underlying data

        :rtype: string
        '''
        if hasattr(self._units, '__iter__'):
            if len(set(self._units)) > 1:
                return self._units
            else:
                return self._units[0]
        else:
            return self._units

    @units.setter
    def units(self, unit):
        self._units = unit
        if self.variables is not None:
            for v in self.variables:
                v.units = unit

    @property
    def varnames(self):
        '''
        Names of underlying variables

        :rtype: [] of strings
        '''
        return [v.varname if hasattr(v, 'varname') else v.name for v in self.variables]

    @property
    def data_shape(self):
        if self.variables is not None:
            return self.variables[0].data.shape
        else:
            return None

    def _get_hash(self, points, time):
        """
        Returns a SHA1 hash of the array of points passed in
        """
        return (hashlib.sha1(points.tobytes()).hexdigest(),
                hashlib.sha1(str(time).encode('utf-8')).hexdigest())

    def _memoize_result(self, points, time, result, D, _copy=True, _hash=None):
        if _copy:
            result = result.copy()
        result.setflags(write=False)
        if _hash is None:
            _hash = self._get_hash(points, time)
        if D is not None and len(D) > 8:
            D.popitem(last=False)
        D[_hash] = result

    def _get_memoed(self, points, time, D, _copy=True, _hash=None):
        if _hash is None:
            _hash = self._get_hash(points, time)
        if (D is not None and _hash in D):
            return D[_hash].copy() if _copy else D[_hash]
        else:
            return None

    def at(self, points, time, units=None, extrapolate=False, memoize=True, _hash=None, _auto_align=True, **kwargs):
        pts = _reorganize_spatial_data(points)
        mem = memoize
        if hash is None:
            _hash = self._get_hash(points, time)

        if mem:
            res = self._get_memoed(points, time, self._result_memo, _hash=_hash)
            if res is not None:
                return res

        value = np.column_stack([var.at(points=points,
                                        time=time,
                                        units=units,
                                        extrapolate=extrapolate,
                                        memoize=memoize,
                                        _hash=_hash,
                                        **kwargs) for var in self.variables])

        if _auto_align == True:
            value = _align_results_to_spatial_data(value.copy(), points)

        if mem:
            self._memoize_result(points, time, value, self._result_memo, _hash=_hash)

        return value

    @classmethod
    def _get_shared_vars(cls, *sh_args):
        default_shared = ['dataset', 'data_file', 'grid_file', 'grid']
        if len(sh_args) != 0:
            shared = sh_args
        else:
            shared = default_shared

        def getvars(func):
            @wraps(func)
            def wrapper(*args, **kws):
                def _mod(n):
                    k = kws
                    s = shared
                    return (n in s) and ((n not in k) or (n in k and k[n] is None))

                if 'filename' in kws and kws['filename'] is not None:
                    kws['data_file'] = kws['grid_file'] = kws['filename']
                ds = dg =  None
                if _mod('dataset'):
                    if 'grid_file' in kws and 'data_file' in kws:
                        if kws['grid_file'] == kws['data_file']:
                            ds = dg = get_dataset(kws['grid_file'])
                        else:
                            ds = get_dataset(kws['data_file'])
                            dg = get_dataset(kws['grid_file'])
                    kws['dataset'] = ds
                else:
                    if 'grid_file' in kws and kws['grid_file'] is not None:
                        dg = get_dataset(kws['grid_file'])
                    else:
                        dg = kws['dataset']
                    ds = kws['dataset']
                if _mod('grid'):
                    gt = kws.get('grid_topology', None)
                    kws['grid'] = Grid.from_netCDF(kws['filename'], dataset=dg, grid_topology=gt)
                if kws.get('varnames', None) is None:
                    varnames = cls._gen_varnames(kws['data_file'],
                                                 dataset=ds)
#                 if _mod('time'):
#                     time = Time.from_netCDF(filename=kws['data_file'],
#                                             dataset=ds,
#                                             varname=data)
#                     kws['time'] = time
                return func(*args, **kws)
            return wrapper
        return getvars
