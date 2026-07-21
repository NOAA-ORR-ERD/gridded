import collections
import logging
import os
from abc import ABC
from functools import wraps
from textwrap import dedent
import time

import netCDF4 as nc4
import numpy as np

from gridded import VALID_LOCATIONS
from gridded.variableapi import VariableAPI
from gridded.depth import Depth
from gridded.grids import Grid, Grid_R, Grid_S, Grid_U
from gridded.time import Time
from gridded.utilities import (
    asarraylike,
    get_dataset,
    parse_filename_dataset_args,
)

log = logging.getLogger(__name__)

class Variable(VariableAPI):
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
    _instance_count = 0

    _default_component_types = {"time": Time, "grid": Grid, "depth": Depth}

    def __init__(
        self,
        name=None,
        units=None,
        time=None,
        data=None,
        grid=None,
        depth=None,
        data_file=None,
        grid_file=None,
        varname=None,
        fill_value=np.nan,
        location=None,
        attributes=None,
        **kwargs,
    ):
        """
        This class represents a value defined on a grid

        :param name: Name
        :type name: string

        :param units: Units
        :type units: string

        :param time: Time axis of the data
        :type time: list of `datetime.datetime`, netCDF4 Variable, or Time object

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
        :type attributes: dict
        """

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
        self.time = time
        self.data_file = data_file
        # the "main" filename for a Variable should be the grid data.
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
            pass  # so just use what was passed in.
        super().__init__(**kwargs)

    #         for k in kwargs:
    #             setattr(self, k, kwargs[k])

    @classmethod
    def from_netCDF(
        cls,
        filename=None,
        varname=None,
        grid_topology=None,
        name=None,
        units=None,
        time=None,
        time_origin=None,
        displacement=None,
        tz_offset=None,
        grid=None,
        depth=None,
        dataset=None,
        data_file=None,
        grid_file=None,
        location=None,
        load_all=False,  # Do we need this? I think not --- maybe a method to fully load later if wanted.
        fill_value=0,
        **kwargs,
    ):
        """
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

        :param tz_offset: offset to compensate for time zone shifts
        :type tz_offset: `datetime.timedelta` or float or integer hours

        :param origin: shifts the time interval to begin at the time specified
        :type origin: `datetime.datetime`

        :param displacement: displacement to apply to the time data.
               Allows shifting entire time interval into future or past
        :type displacement: `datetime.timedelta`
        """

        Grid = cls._default_component_types["grid"]
        Time = cls._default_component_types["time"]
        Depth = cls._default_component_types["depth"]
        if filename is not None:
            data_file = filename
            grid_file = filename

        ds, dg = parse_filename_dataset_args(
            filename=filename, dataset=dataset, grid_file=grid_file, data_file=data_file
        )

        if grid is None:
            grid = Grid.from_netCDF(grid_file, dataset=dg, grid_topology=grid_topology)
        if varname is None:
            varname = cls._gen_varname(data_file, dataset=ds)
            if varname is None:
                raise NameError("Default current names are not in the data file, must supply variable name")
        data = ds.variables[varname]
        if name is None:
            name = cls.__name__ + "_" + str(cls._instance_count)
            cls._instance_count += 1
        if units is None:
            try:
                units = data.units
            except AttributeError:
                units = None

        if time is None:
            timevarname = Time.locate_time_var_from_var(data)
            if timevarname is None:
                time = Time()
            else:
                time = Time.from_netCDF(
                    filename=data_file,
                    dataset=ds,
                    varname=timevarname,
                    # datavar=None,
                    tz_offset=tz_offset,
                    new_tz_offset=None,
                    origin=time_origin,
                    displacement=displacement,
                )
        else:
            timevarname = 1 if len(time) > 1 else 0

        if depth is None:
            istimevar = 0 if timevarname is None else 1

            if (
                isinstance(grid, (Grid_S, Grid_R))
                and len(data.shape) == 3 + istimevar
                or isinstance(grid, Grid_U)
                and len(data.shape) == 2 + istimevar
            ):
                depth = Depth.from_netCDF(grid_file=dg, dataset=ds, time=time, grid=grid, **kwargs)
        if location is None:
            if hasattr(data, "location"):
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
        return cls(
            name=name,
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
            **kwargs,
        )

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return (
            f"{self.__class__.__module__}.{self.__class__.__name__}("
            f'name="{self.name}", \n'
            f'time="{self.time}", \n'
            f'units="{self.units}", \n'
            f'location="{self.location}" \n'
            f'data=Type:{type(self.data)}, shape:{self.data.shape}", '
            ")"
        )

    @classmethod
    def constant(cls, value, **kwargs):
        # Sets a Variable up to represent a constant scalar field. The result
        # will return a constant value for all times and places.
        _data = np.asarray(value)
        return cls(data=_data,**kwargs)

    @property
    def location(self):
        if self._location is None and self.data is not None and hasattr(self.data, "location"):
            return self.data.location
        else:
            return self._location

    @location.setter
    def location(self, location):
        # Fixme: perhaps we need Variable subclasses,
        #        to distingish between variable types.
        if location not in VALID_LOCATIONS:
            raise ValueError(f"Invalid location: {location}, must be one of: {VALID_LOCATIONS}")
        self._location = location

    @property
    def info(self):
        """
        Information about the variable object
        This could be filled out more
        """
        try:
            std_name = self.attributes["standard_name"]
        except KeyError:
            std_name = None
        msg = f"""
              Variable:
                filename: {self.filename}
                varname: {self.varname}
                standard name: {std_name}
                units: {self.units}
                grid: {self.grid}
                data shape: {self.data.shape}
              """
        return dedent(msg)

    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, t):
        Time_class = self.__class__._default_component_types["time"]
        if t is None:
            self._time = None
            return
        if isinstance(t, Time_class):
            self._time = t
        elif isinstance(t, collections.abc.Iterable) or isinstance(t, nc4.Variable):
            self._time = Time_class(t)
        else:
            raise ValueError("Time must be set with an iterable container or netCDF variable")

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, d):
        d = asarraylike(d)
        self._data = d

    @property
    def units(self):
        """
        Units of underlying data

        :rtype: string
        """
        return self._units

    @units.setter
    def units(self, unit):
        #         if unit is not None:
        #             if not unit_conversion.is_supported(unit):
        #                 raise ValueError('Units of {0} are not supported'.format(unit))
        self._units = unit
    
    def _compute_at(self, ps, ts, extrapolate=None, unmask=False, _hash=None):
        """
        Computation core of the .at function. All arguments should be prepared for internal use.
        Note that unit conversions are not explicitly handled in this function 
        (though may occur if subcomponent Variable are present)
        
        Returns the interpolated values at the points and time specified.
        """
        
        order = self.dimension_ordering
        if len(order) == 0:
            #special case for a Variable with no dimensions (eg a constant value)
            return np.full((len(ps), len(ts), 1), self.data)
        
        if self.memoization_enabled:
            res = self._get_memoed(ps, ts, self._result_memo, _hash=_hash)
            if res is not None:
                return res
        
        _mem = self.memoization_enabled
        
        retval = np.ma.empty((len(ps), len(ts), 1))
        for i, time in enumerate(ts):
            if order[0] == "time":
                value = self._time_interp(ps, time, extrapolate=extrapolate, _mem=_mem, _hash=_hash)
            elif order[0] == "depth":
                value = self._depth_interp(ps, time, extrapolate=extrapolate, _mem=_mem, _hash=_hash)
            else:
                value = self._xy_interp(ps, time, extrapolate=extrapolate, _mem=_mem, _hash=_hash)
            value = value.reshape(-1, 1)
            retval[:, i, :] = value
        return retval

    def _xy_interp(self, points, time, extrapolate=None, slices=(), _mem=False, _hash=None):
        """
        Uses the py(s/u)grid interpolation to determine the values at the points, and returns it
        :param points: Coordinates to be queried (3D)
        :type points: Nx3 array of double

        :param time: Time of the query
        :type time: datetime.datetime object

        :param extrapolate: Turns extrapolation on or off
        :type extrapolate: boolean

        :param slices: describes how the data needs to be sliced to reach the appropriate dimension
        :type slices: tuple of integers or slice objects
        """

        value = self.grid.interpolate_var_to_points(
            points[:, 0:2],
            self.data,
            location=self.location,
            extrapolate=extrapolate,
            slices=slices,
            _hash=self._get_hash(points[:, 0:2], time, extrapolate), #grid uses only lon/lat + time for memoization (for now)
            _memo=_mem,
        )
        return value

    def _time_interp(self, points, time, slices=(), extrapolate=False, _mem=False, _hash=None):
        """
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
        """
        order = self.dimension_ordering
        idx = order.index("time")
        if idx == len(order) - 1:
            # time is the last dimension, so directly interpolate the data
            ind, alpha = self.time.interpolation_alpha(time, extrapolate=extrapolate)
            s0 = slices + (ind - 1,)
            s1 = slices + (ind,)
            v0 = self.data[s0]
            v1 = self.data[s1]
            return v0 + (v1 - v0) * alpha
        
        if order[idx + 1] != "depth":
            val_func = self._xy_interp
            vf_kwargs = grid_kwargs
        else:
            val_func = self._depth_interp
            vf_kwargs = depth_kwargs

        if time == self.time.min_time or (extrapolate and time < self.time.min_time):
            # min or before
            return val_func(points, time, slices=(0,), extrapolate=extrapolate, _mem=_mem, _hash=_hash)
        elif time == self.time.max_time or (extrapolate and time > self.time.max_time):
            return val_func(points, time, slices=(-1,), extrapolate=extrapolate, _mem=_mem, _hash=_hash)
        else:
            ind, alpha = self.time.interpolation_alpha(time, extrapolate=extrapolate)
            s0 = slices + (ind - 1,)
            s1 = slices + (ind,)
            v0 = val_func(points, time, slices=s0, extrapolate=extrapolate, _mem=_mem, _hash=_hash)
            v1 = val_func(points, time, slices=s1, extrapolate=extrapolate, _mem=_mem, _hash=_hash)

            return v0 + (v1 - v0) * alpha

    def _depth_interp(self, points, time, slices=(), extrapolate=False, _mem=False, _hash=None):
        """
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
        """
        order = self.dimension_ordering
        dim_idx = order.index("depth")
        if dim_idx == len(order) - 1:
            # depth is the last dimension, so directly interpolate the data
            ind, alpha = self.depth.interpolation_alpha(points, time, extrapolate=extrapolate)
            s0 = slices + (ind - 1,)
            s1 = slices + (ind,)
            v0 = self.data[s0]
            v1 = self.data[s1]
            return v0 + (v1 - v0) * alpha
        
        if order[dim_idx + 1] != "time":
            val_func = self._xy_interp
            vf_kwargs = grid_kwargs
        else:
            val_func = self._time_interp
            vf_kwargs = time_kwargs

        d_indices, d_alphas = self.depth.interpolation_alphas(
            points,
            time,
            (self.depth.num_levels,), #level interpolation only
            extrapolate=extrapolate,
            _mem=_mem,
            _hash=_hash
        )

        if isinstance(d_indices, np.ma.MaskedArray) and np.all(d_indices.mask):
            # all particles ended up masked
            rtv = np.empty((points.shape[0],), dtype=np.float64) * np.nan
            rtv = np.ma.MaskedArray(data=rtv, mask=np.full(rtv.shape, True))
            return rtv

        msk = np.isnan(d_indices) if not np.ma.is_masked(d_indices) else d_indices.mask
        values = np.ma.MaskedArray(data=np.empty((points.shape[0],)) * np.nan, mask=msk)
        # Points are mixed within the grid. Some may be above the surface or under the ground
        uniq_idx = np.unique(d_indices)
        if np.ma.masked in uniq_idx:  # the [0:-1] is required to skip all masked indices
            uniq_idx = uniq_idx[0:-1]
        for idx in uniq_idx:
            lay_idxs = np.where(d_indices == idx)[0]
            if self.data.shape[dim_idx] == self.depth.num_layers:
                #data is located on cell centers, so value is held constant across cell.
                if idx == -1:
                    #special case for out of bounds and extrapolation.
                    idx = 0
                values.put(lay_idxs, val_func(points[lay_idxs], time, extrapolate, slices=slices + (idx,), **kwargs))
            else:
                #data is on cell edge, or on nodes, so we need to interpolate between the two levels.
                if idx == self.data.shape[dim_idx] - 1:
                    # special case, index == depth dim length, so only v0 is valid
                    v0 = val_func(points[lay_idxs], time, extrapolate, slices=slices + (idx,), _mem=_mem, _hash=_hash)
                    values.put(lay_idxs, v0)
                    continue

                v1 = val_func(points[lay_idxs], time, extrapolate, slices=slices + (idx + 1,), _mem=_mem, _hash=_hash)
                v0 = val_func(points[lay_idxs], time, extrapolate, slices=slices + (idx,), _mem=_mem, _hash=_hash)
                sub_vals = v0 + (v1 - v0) * d_alphas[lay_idxs]
                # partially fill the values array for this layer
                values.put(lay_idxs, sub_vals)
        return values

    @classmethod
    def _gen_varname(cls, filename=None, dataset=None, names_list=None, std_names_list=None):
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
                if (
                    hasattr(var, "standard_name")
                    and var.standard_name == n
                    or hasattr(var, "long_name")
                    and var.long_name == n
                ):
                    return var.name
        raise ValueError("Default names not found.")


class VectorVariable(VariableAPI):
    # Fixme: a lot of code duplication in here

    # Keys are component names ('u', 'v', etc) and values are the netCDF4 names.
    # eg {'u': ['u', 'U', 'eastward_sea_water_velocity']}
    default_names = {}

    # Keys are component names ('u', 'v', etc) and values are the CF names.
    # eg {'u': ['u', 'U', 'eastward_sea_water_velocity']}
    cf_names = {}

    # This list of strings specify names for each component of the vector variable.
    # The names should be the same as keys in default_names and cf_names
    # for example, ['u', 'v'] will allow vv.u and vv.v to be used to access the components
    # instead of vv.variables[0] and vv.variables[1]
    # An error will raise if comp_order is longer than the number of components (vv.variables)
    # upon object initialization
    comp_order = []

    _instance_count = 0

    """'
    These are the classes which are used when internal components are created
    by default, such as automatically from a file or other python data structure.
    Subclasses of this type may override this to use different classes for it's
    components
    """
    _default_component_types = {"time": Time, "grid": Grid, "depth": Depth, "variable": Variable}

    def __init__(
        self,
        name=None,
        units=None,
        time=None,
        variables=None,
        grid=None,
        depth=None,
        grid_file=None,
        data_file=None,
        fill_value=np.nan,
        **kwargs,
    ):

        super().__init__()
        self.name = self._units = self._time = self._variables = None

        self.name = name

        if all([isinstance(v, Variable) for v in variables]):
            if time is not None and not isinstance(time, Time):
                time = Time(time)
            units = variables[0].units if units is None else units
            time = variables[0].time if time is None else time
        self._time = time
        if units is None:
            units = variables[0].units
        self._units = units
        self.fill_value=fill_value
        if variables is None or len(variables) < 2:
            raise ValueError("Variables must be an array-like of 2 or more Variable objects")
        self.variables = variables
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
    def from_netCDF(
        cls,
        filename=None,
        varnames=None,
        grid_topology=None,
        name=None,
        units=None,
        time=None,
        time_origin=None,
        displacement=None,
        tz_offset=None,
        grid=None,
        depth=None,
        data_file=None,
        grid_file=None,
        dataset=None,
        load_all=False,
        variables=None,
        **kwargs,
    ):
        """
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

        :param tz_offset: offset to compensate for time zone shifts
        :type tz_offset: `datetime.timedelta` or float or integer hours

        :param origin: shifts the time interval to begin at the time specified
        :type origin: `datetime.datetime`

        :param displacement: displacement to apply to the time data.
               Allows shifting entire time interval into future or past
        :type displacement: `datetime.timedelta`

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
        """
        Grid = cls._default_component_types["grid"]
        Time = cls._default_component_types["time"]
        Variable = cls._default_component_types["variable"]
        Depth = cls._default_component_types["depth"]
        if filename is not None:
            data_file = filename
            grid_file = filename

        ds, dg = parse_filename_dataset_args(
            filename=filename, dataset=dataset, grid_file=grid_file, data_file=data_file
        )

        if grid is None:
            grid = Grid.from_netCDF(grid_file, dataset=dg, grid_topology=grid_topology)
        if varnames is None:
            varnames = cls._gen_varnames(data_file, dataset=ds)
            if all([v is None for v in varnames]):
                raise ValueError("No compatible variable names found!")
        if name is None:
            name = cls.__name__ + "_" + str(cls._instance_count)
            cls._instance_count += 1
        data = ds[varnames[0]]

        if time is None:
            timevarname = Time.locate_time_var_from_var(data)
            if timevarname is None:
                time = Time()
            else:
                time = Time.from_netCDF(
                    filename=data_file,
                    dataset=ds,
                    varname=timevarname,
                    # datavar=None,
                    tz_offset=tz_offset,
                    new_tz_offset=None,
                    origin=time_origin,
                    displacement=displacement,
                )
        else:
            timevarname = 1 if len(time) > 1 else 0

        if depth is None:
            istimevar = 0 if timevarname is None else 1

            if (
                isinstance(grid, (Grid_S, Grid_R))
                and len(data.shape) == 3 + istimevar
                or isinstance(grid, Grid_U)
                and len(data.shape) == 2 + istimevar
            ):
                depth = Depth.from_netCDF(grid_file=dg, dataset=ds, time=time, grid=grid, **kwargs)

        if variables is None:
            variables = []
            for vn in varnames:
                if vn is not None:
                    variables.append(
                        Variable.from_netCDF(
                            filename=filename,
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
                            **kwargs,
                        )
                    )
        if units is None:
            units = [v.units for v in variables]
            if all(u == units[0] for u in units):
                units = units[0]
        return cls(
            name=name,
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
            **kwargs,
        )

    @classmethod
    def _gen_varnames(cls, filename=None, dataset=None, names_dict=None, std_names_dict=None):
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
                        if (
                            hasattr(var, "standard_name")
                            and var.standard_name == n
                            or hasattr(var, "long_name")
                            and var.long_name == n
                        ):
                            rd[k] = var.name
                            break
        return collections.namedtuple("varnames", cls.comp_order)(**rd)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return (
            f"{self.__class__.__module__}.{self.__class__.__name__}("
            f'name="{self.name}", '
            f'time="{self.time}", '
            f'units="{self.units}", '
            f'variables="{self.variables}", '
            f'grid="{self.grid}", '
            ")"
        )

    @property
    def location(self):
        return [v.location for v in self.variables]

    locations = location

    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, t):
        Time_class = self.__class__._default_component_types["time"]
        if self.variables is not None:
            for v in self.variables:
                try:
                    v.time = t
                except ValueError as e:
                    raise ValueError(f"""Time was not compatible with variables.
                    Set variables attribute to None to allow changing other attributes
                    Original error: {str(e)}""")
        if isinstance(t, Time_class):
            self._time = t
        elif isinstance(t, collections.Iterable) or isinstance(t, nc4.Variable):
            self._time = Time_class(t)
        else:
            raise ValueError("Time must be set with an iterable container or netCDF variable")

    @property
    def units(self):
        """
        Units of underlying data

        :rtype: string
        """
        if hasattr(self._units, "__iter__"):
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
        """
        Names of underlying variables

        :rtype: [] of strings
        """
        return [v.varname if hasattr(v, "varname") else v.name for v in self.variables]

    @property
    def data_shape(self):
        if self.variables is not None:
            return self.variables[0].data.shape
        else:
            return None
    
    def _compute_at(self, ps, ts, extrapolate=False, unmask=False, _hash=None):
        """
        Computation core of the .at function. All arguments should be prepared for internal use.
        Note that unit conversions are not explicitly handled in this function 
        (though may occur if subcomponent Variable are present)
        
        Returns the interpolated values at the points and time specified.
        """
        if self.memoization_enabled:
            res = self._get_memoed(ps, ts, self._result_memo, _hash=_hash)
            if res is not None:
                return res
        value = np.ma.dstack(
            [
                var.at(
                    ps,
                    ts,
                    extrapolate=extrapolate,
                    unmask=unmask,
                    _hash=_hash,
                )
                for var in self.variables
            ]
        )
        return value
    
    @classmethod
    def _get_shared_vars(cls, *sh_args):
        default_shared = ["dataset", "data_file", "grid_file", "grid"]
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

                if "filename" in kws and kws["filename"] is not None:
                    kws["data_file"] = kws["grid_file"] = kws["filename"]
                ds = dg = None
                if _mod("dataset"):
                    if "grid_file" in kws and "data_file" in kws:
                        if kws["grid_file"] == kws["data_file"]:
                            ds = dg = get_dataset(kws["grid_file"])
                        else:
                            ds = get_dataset(kws["data_file"])
                            dg = get_dataset(kws["grid_file"])
                    kws["dataset"] = ds
                else:
                    if "grid_file" in kws and kws["grid_file"] is not None:
                        dg = get_dataset(kws["grid_file"])
                    else:
                        dg = kws["dataset"]
                    ds = kws["dataset"]
                if _mod("grid"):
                    gt = kws.get("grid_topology", None)
                    kws["grid"] = Grid.from_netCDF(kws["filename"], dataset=dg, grid_topology=gt)
                if kws.get("varnames", None) is None:
                    varnames = cls._gen_varnames(kws["data_file"], dataset=ds)
                #                 if _mod('time'):
                #                     time = Time.from_netCDF(filename=kws['data_file'],
                #                                             dataset=ds,
                #                                             varname=data)
                #                     kws['time'] = time
                return func(*args, **kws)

            return wrapper

        return getvars

    def save(self, filepath, format="netcdf4"):
        """
        Save the variable object to a netcdf file.

        :param filepath: path to file you want o save to. or a writable
                         netCDF4 Dataset An existing one
                         If a path, an existing file will be clobbered.
        :type filepath: string

        Follows the convention established by the netcdf UGRID working group:

        http://ugrid-conventions.github.io/ugrid-conventions

        """
        format_options = ("netcdf3", "netcdf4")
        if format not in format_options:
            raise ValueError(f"format: {format} not supported. Options are: {format_options}")
