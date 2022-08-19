from __future__ import absolute_import, division, print_function, unicode_literals

from textwrap import dedent
import collections
import hashlib
from functools import wraps
import os

import numpy as np
import netCDF4 as nc4

from gridded.utilities import (get_dataset,
                               _reorganize_spatial_data,
                               _align_results_to_spatial_data,
                               asarraylike)
from gridded import VALID_LOCATIONS
from gridded.grids import Grid, Grid_U, Grid_S, Grid_R
from gridded.depth import Depth
from gridded.time import Time

import logging

log = logging.getLogger(__name__)


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
                 location=None,
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
        :type grid: Grid object (pysgrid or pyugrid or )

        :param data_file: Name of data source file
        :type data_file: string

        :param grid_file: Name of grid source file
        :type grid_file: string

        :param varname: Name of the variable in the data source file
        :type varname: string

        :param fill_value: the fill value used for undefined data

        :param location: location on the grid -- possible values
                         depend on the grid type
        :type location: str

        :param attributes: attributes associated with the Variable
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
        self.location = location
        self.data = data
        self.time = time if time is not None else self._default_component_types['time'].constant_time()
        self.data_file = data_file
        # the "main" filename for a Varibale should be the grid data.
        self.filename = data_file
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
                    location=None,
                    load_all=False,  # Do we need this? I think not --- maybe a method to fully load later if wanted.
                    fill_value=0,
                    **kwargs
                    ):
        '''
        Allows one-function creation of a Variable from a file.

        :param filename: Default data source. Has lowest priority.
                         If dataset, grid_file, or data_file are provided,
                         this function uses them first
        :type filename: string

        :param varname: Explicit name of the data in the data source file.
                        Equivalent to the key used to look the item up
                        directly eg 'ds["lon_u"]' for a netCDF4 Dataset.
        :type varname: string

        :param grid_topology: Description of the relationship between grid
                              attributes and variable names.
        :type grid_topology: {string : string, ...}

        :param name: Name of this object
        :type name: string

        :param units: string such as 'm/s'
        :type units: string

        :param time: Time axis of the data. May be a constructed ``gridded.Time``
                     object, or collection of datetime.datetime objects
        :type time: [] of datetime.datetime, netCDF4 Variable, or Time object

        :param data: Underlying data object. May be any array-like,
                     including netCDF4 Variable, etc
        :type data: netCDF4.Variable or numpy.array

        :param grid: Grid that the data corresponds to
        :type grid: pysgrid or pyugrid

        :param location: The feature where the data aligns with the grid.
        :type location: string

        :param depth: Depth axis object from ``gridded.depth``
        :type depth: Depth, S_Depth or L_Depth

        :param dataset: Instance of open netCDF4.Dataset
        :type dataset: netCDF4.Dataset

        :param data_file: Name of data source file, if data and grid files are separate
        :type data_file: string

        :param grid_file: Name of grid source file, if data and grid files are separate
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
        if data_file is None:
            data_file = os.path.split(ds.filepath())[-1]

        if grid is None:
            grid = Grid.from_netCDF(grid_file,
                                    dataset=dg,
                                    grid_topology=grid_topology)
        if varname is None:
            varname = cls._gen_varname(data_file,
                                       dataset=ds)
            if varname is None:
                raise NameError('Default current names are not in the data file, '
                                'must supply variable name')
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
                time = Time(data=time.data,
                            filename=time.filename,
                            varname=time.varname,
                            origin=time_origin)
        if depth is None:
            if (isinstance(grid, (Grid_S, Grid_R)) and len(data.shape) == 4 or
                    isinstance(grid, Grid_U) and len(data.shape) == 3):
                depth = Depth.from_netCDF(grid_file=dg,
                                          data_file=ds,
                                          **kwargs
                                          )
        if location is None:
            if hasattr(data, 'location'):
                location = data.location
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
                   location=location,
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
    def location(self):
        if self._location is None and self.data is not None and hasattr(self.data, 'location'):
            return self.data.location
        else:
            return self._location

    @location.setter
    def location(self, location):
        # Fixme: perhaps we need Variable subclasses,
        #        to distingish between variable types.
        if location not in VALID_LOCATIONS:
            raise ValueError("Invalid location: {}, must be one of: {}".format(location, VALID_LOCATIONS))
        self._location = location

    @property
    def info(self):
        """
        Information about the variable object
        This could be filled out more
        """
        try:
            std_name = self.attributes['standard_name']
        except KeyError:
            std_name = None
        msg = """
              Variable:
                filename: {0.filename}
                varname: {0.varname}
                standard name: {1}
                units: {0.units}
                grid: {0.grid}
                data shape: {0.data.shape}
              """.format(self, std_name)
        return dedent(msg)

    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, t):
        Time_class = self.__class__._default_component_types['time']
        if t is None:
            self._time = None
            return
        if self.data is not None and len(t) != self.data.shape[0] and len(t) > 1:
            raise ValueError("Data/time interval mismatch")
        if isinstance(t, Time_class):
            self._time = t
        elif isinstance(t, collections.Iterable) or isinstance(t, nc4.Variable):
            self._time = Time_class(t)
        else:
            raise ValueError("Time must be set with an iterable container or netCDF variable")

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, d):
        d = asarraylike(d)
        # Fixme: maybe all this checking should be done when it gets added to the Dataset??
        if self.time is not None and len(d) != len(self.time):
            raise ValueError("Data/time interval mismatch")
        ## fixme: we should check Depth, too.
        # if self.grid is not None and self.grid.infer_location(d) is None:
        #     raise ValueError("Data/grid shape mismatch. Data shape is {0}, Grid shape is {1}".format(d.shape, self.grid.node_lon.shape))
        if self.grid is not None:  # if there is not a grid, we can't check this
            if self.location is None:  # not set, let's try to figure it out
                self.location = self.grid.infer_location(d)
            if self.location is None:
                raise ValueError("Data/grid shape mismatch: Data shape is {0}, "
                                 "Grid shape is {1}".format(d.shape, self.grid.node_lon.shape))
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
        """
        interpolate data to the center of the cells

        :param time: the time to interpolate at

        **Warning:** NOT COMPLETE

        NOTE: what if this data is already on the cell centers?
        """
        raise NotImplementedError("center_values is not finished")

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
        Returns a list that describes the dimensions of the property's data.
        If a dimension_ordering is assigned, it will continue to use that.
        If no dimension_ordering is set, then a default ordering will be generated
        based on the object properties and data shape.

        For example, if the data has 4 dimensions and is represented by a
        Grid_S (structured grid), and the Variable has a depth and time assigned,
        then the assumed ordering is ['time','depth','lon','lat']

        If the data has 3 dimensions, self.grid is a Grid_S, and self.time is None,
        then the ordering is ['depth','lon','lat']
        If the data has 3 dimensions, self.grid is a Grid_U, the ordering is
        ['time','depth','ele']
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
                    raise ValueError('Generated ordering too short to fit data. '
                                     'Time or depth must not be None')
            elif diff == 2:
                order = order[2:]
            else:
                raise ValueError('Too many/too few dimensions ndim={0}'.format(ndim))
            return order

    @dimension_ordering.setter
    def dimension_ordering(self, order):
        self._order = order

#     @profile
    def at(self,
           points,
           time,
           units=None,
           extrapolate=False,
           _hash=None,
           _mem=True,
           _auto_align=True,
           unmask=False,
           **kwargs):
        """
        Find the value of the property at positions P at time T

        :param points: Cartesian coordinates to be queried (P).
        [[Lon1, Lat1, Z1], [Lon2, Lat2, Z2], ...]
        Note that if your Z is positive-up, self.depth.positive_down should be
        set to False
        :type points: Nx3 array of double

        :param time: The time at which to query these points (T)
        :type time: datetime.datetime object

        :param units: units the values will be returned in (or converted to)
        :type units: string such as ('m/s', 'knots', etc)

        :param extrapolate: if True, extrapolation will be supported
        :type extrapolate: boolean (default False)

        :param unmask: if True and return array is a masked array, returns filled array
        :type unmask: boolean (default False)

        :param zero_ref: Specifies whether the zero datum moves with zeta or not. Only
        applicable if depth dimension is present with full sigma layers
        :type zero_ref: string 'absolute' or 'relative'

        :return: returns a Nx1 array of interpolated values
        :rtype: double
        """
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

    interpolate = at #common request

    def _xy_interp(self, points, time, extrapolate, slices=(), **kwargs):
        '''
        Uses the py(s/u)grid interpolation to determine the values at the points, and returns it
        :param points: Coordinates to be queried (3D)
        :type points: Nx3 array of double

        :param time: Time of the query
        :type time: datetime.datetime object

        :param extrapolate: Turns extrapolation on or off
        :type extrapolate: boolean

        :param slices: describes how the data needs to be sliced to reach the appropriate dimension
        :type slices: tuple of integers or slice objects
        '''
        _hash = kwargs['_hash'] if '_hash' in kwargs else None
        units = kwargs['units'] if 'units' in kwargs else None

        value = self.grid.interpolate_var_to_points(points[:, 0:2],
                                                    self.data,
                                                    location=self.location,
                                                    _hash=self._get_hash(points[:, 0:2],
                                                                         time),
                                                    slices=slices, _memo=True)
        return value

    def _time_interp(self, points, time, extrapolate, slices=(), **kwargs):
        '''
        Uses the Time object to interpolate the result of the next level of interpolation, as specified
        by the dimension_ordering attribute.
        :param points: Coordinates to be queried (3D)
        :type points: Nx3 array of double

        :param time: Time of the query
        :type time: datetime.datetime object

        :param extrapolate: Turns extrapolation on or off
        :type extrapolate: boolean

        :param slices: describes how the data needs to be sliced to reach the appropriate dimension
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
            alphas = self.time.interp_alpha(time, extrapolate)
            value = v0 + (v1 - v0) * alphas
            return value

    def _depth_interp(self, points, time, extrapolate, slices=(), **kwargs):
        '''
        Uses the Depth object to interpolate the result of the next level of interpolation, as specified
        by the dimension_ordering attribute.
        :param points: Coordinates to be queried (3D)
        :type points: Nx3 array of double

        :param time: Time of the query
        :type time: datetime.datetime object

        :param extrapolate: Turns extrapolation on or off
        :type extrapolate: boolean

        :param slices: describes how the data needs to be sliced to reach the appropriate dimension
        :type slices: tuple of integers or slice objects
        '''
        order = self.dimension_ordering
        idx = order.index('depth')
        if order[idx + 1] != 'time':
            val_func = self._xy_interp
        else:
            val_func = self._time_interp
        d_indices, d_alphas = self.depth.interpolation_alphas(points, time, self.data.shape[1:], _hash=kwargs.get('_hash', None), extrapolate=extrapolate)
        # the index of a point is -1 if it is on the surface
        # the index refers to the layer index ABOVE the point. EG index 5 means v0 should
        # be from layer 4 and v1 should be from layer 5
        # HOWEVER if the depth layer data is "backwards" for some reason ('high-index-bottom'),
        # (self.bottom_index > self.surface_index) then index 5 means v0 should be from
        # layer 6 and v1 should be from layer 5
        if d_indices is None and d_alphas is None:
            # all particles are on surface
            return val_func(points, time, extrapolate, slices=slices + (self.depth.surface_index,), **kwargs)
        elif np.all(d_indices == -1) and np.all(d_alphas == -2):
            # all particles underground
            return val_func(points, time, extrapolate, slices=slices + (self.depth.bottom_index,), **kwargs)
        else:
            direction = 1 #bottom idx < top
            underwater = d_indices != -1
            if self.depth.bottom_index > self.depth.surface_index:
                low_layer = d_indices[underwater].max()
                high_layer = d_indices.min() - 1
                direction = -1
            else:
                low_layer = d_indices[underwater].min()
                high_layer = d_indices.max() + 1
            values = np.zeros(len(points), dtype=np.float64)
            v0_idx = low_layer - direction
            lay_idxs = np.where(d_indices == low_layer)[0]
            v0 = val_func(points[lay_idxs], time, extrapolate, slices=slices + (v0_idx,), **kwargs)
            # if we get an array-out-of bounds on the line above it would be because the index
            # array had the max index (n_layers) in a high-index-bottom situation.
            # regardless, this means depth.interpolation_alphas did something wrong because
            # such values should not be introduced.
            for idx in range(low_layer, high_layer, direction):
                #select the matching LEs for this layer
                lay_idxs = np.where(d_indices == low_layer)[0]
                if len(lay_idxs) == 0:
                    #no point indices are within the layer, so skip
                    continue
                if v0_idx != idx - 1:
                    #skipped one or more layers, so new v0 needs to be found
                    v0_idx = idx - 1
                    v0 = val_func(points[lay_idxs], time, extrapolate, slices=slices + (v0_idx,), **kwargs)
                v1 = val_func(points[lay_idxs], time, extrapolate, slices=slices + (idx,), **kwargs)
                sub_vals = v0 + (v1 - v0) * d_alphas[lay_idxs]
                #partially fill the values array for this layer
                values.put(lay_idxs, sub_vals)
                #shift up to the next layer
                v0 = v1
                v0_idx = idx
            underground = (d_alphas == -2)
            # These would have been flagged with index -1 (surface)
            # but the additional d_alpha of -2 indicates their subterranean
            # nature. Therefore, use fill_value
            values[underground] = self.fill_value
            return values

    def _transect(self, times, depths, points):
        '''
        returns a transect of the Variable at given values.
        This function is not close to finished.
        '''
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
        Function to find the default variable names if they are not provided. This
        function does nothing without defined default_names or cf_names class
        attributes

        :param filename: Name of file that will be searched for variables
        :type filename: string

        :param dataset: Existing instance of a netCDF4.Dataset
        :type dataset: netCDF.Dataset

        :return: name of first netCDF4.Variable that matches
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

        super(VectorVariable, self).__init__()
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

        self._result_memo = collections.OrderedDict()
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
        :type filename: string

        :param varnames: Names of the variables in the data source file
        :type varnames: [] of string

        :param grid_topology: Description of the relationship between grid attributes and variable names.
        :type grid_topology: {string : string, ...}

        :param name: Name of property
        :type name: string

        :param units: Units
        :type units: string

        :param time: Time axis of the data
        :type time: [] of datetime.datetime, netCDF4 Variable, or Time object

        :param data: Underlying data source
        :type data: netCDF4.Variable or numpy.array

        :param grid: Grid that the data corresponds with
        :type grid: pysgrid or pyugrid

        :param dataset: Instance of open Dataset
        :type dataset: netCDF4.Dataset

        :param data_file: Name of data source file
        :type data_file: string

        :param grid_file: Name of grid source file
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
            grid = Grid.from_netCDF(grid_file=dg,
                                    data_file=ds,
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
                                          **kwargs
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
                    # Fixme: We're calling from_netCDF from itself ?!?!?
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
                                                          location=None,
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
                   location=None,
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
        :type filename: string

        :param dataset: Existing instance of a netCDF4.Dataset
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
    def location(self):
        return [v.location for v in self.variables]

    locations = location

    @property
    def is_data_on_nodes(self):
        return self.grid.infer_location(self.variables[0].data) == 'node'

    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, t):
        Time_class = self.__class__._default_component_types['time']
        if self.variables is not None:
            for v in self.variables:
                try:
                    v.time = t
                except ValueError as e:
                    raise ValueError('''Time was not compatible with variables.
                    Set variables attribute to None to allow changing other attributes
                    Original error: {0}'''.format(str(e)))
        if isinstance(t, Time_class):
            self._time = t
        elif isinstance(t, collections.Iterable) or isinstance(t, nc4.Variable):
            self._time = Time_class(t)
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

    def save(self, filepath, format='netcdf4'):
        """
        Save the variable object to a netcdf file.

        :param filepath: path to file you want o save to. or a writable
                         netCDF4 Dataset An existing one
                         If a path, an existing file will be clobbered.
        :type filepath: string

        Follows the convention established by the netcdf UGRID working group:

        http://ugrid-conventions.github.io/ugrid-conventions

        """
        format_options = ('netcdf3', 'netcdf4')
        if format not in format_options:
            raise ValueError("format: {} not supported. Options are: {}".format(format, format_options))

