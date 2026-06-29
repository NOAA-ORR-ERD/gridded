import os
import warnings

import numpy as np

from gridded.grids import Grid
from gridded.time import Time
from gridded.utilities import (
    can_create_class,
    get_dataset,
    merge_var_search_dicts,
    parse_filename_dataset_args,
    search_dataset_for_any_long_name,
    search_dataset_for_variables_by_longname,
    search_dataset_for_variables_by_varname,
    search_netcdf_vars,
)

class DepthBase:
    _instance_count = 0
    _default_component_types = {"time": Time, "grid": Grid, "variable": None}
    # 'variable' is None here to avoid import issues.
    # It is set in the __init__.py

    def __init__(
        self,
        name=None,
        surface_index=-1,
        bottom_index=0,
        default_surface_boundary_condition="extrapolate",
        default_bottom_boundary_conditon="mask",
        **kwargs,
    ):
        """
        :param surface_index: array index of 'highest' level
                              (closest to sea level)
        :param bottom_index: array index of 'lowest' level
                             (closest to seafloor)
        """
        self.name = name
        self._surface_index = surface_index
        self._bottom_index = bottom_index
        self.default_surface_boundary_condition = (
            default_surface_boundary_condition
        )
        self.default_bottom_boundary_condition = (
            default_bottom_boundary_conditon
        )

    @classmethod
    def _can_create_from_netCDF(
        cls,
        filename=None,
        dataset=None,
        grid_file=None,
        data_file=None,
    ):
        return True

    @classmethod
    def from_netCDF(cls, surface_index=-1, **kwargs):
        return cls(surface_index, **kwargs)

    @property
    def surface_index(self):
        # Subclasses are REQUIRED to implement this property in the manner
        # appropriate for the system being represented
        return self._surface_index

    @property
    def bottom_index(self):
        # Subclasses are REQUIRED to implement this property in the manner
        # appropriate for the system being represented
        return self._bottom_index

    def interpolation_alphas(
        self,
        points,
        time,
        data_shape,
        extrapolate=False,
        _hash=None,
        **kwargs,
    ):
        """
        Returns the weights (alphas) required for interpolation

        The base class implementation only supports surface values:
        it returns the surface index and 0.0 for the alpha

        This is the expected outcome for the case where all points
        are on the surface.
        """
        sz = len(points)
        indices = np.ma.MaskedArray(
            data=np.ones((sz,), dtype=np.int64) * self.surface_index,
            mask=np.zeros((sz,), dtype=bool),
        )
        alphas = np.ma.MaskedArray(
            data=np.zeros((sz,), dtype=np.float64),
            mask=np.zeros((sz,), dtype=bool),
        )
        return indices, alphas

    @classmethod
    def _find_required_depth_attrs(
        cls, filename, dataset=None, depth_topology=None
    ):
        """
        This function is the top level 'search for attributes' function.
        If there are any common attributes to all potential depth types,
        they will be sought here.

        This function returns a dict, which maps an attribute name to a
        netCDF4 Variable or numpy array object extracted from the dataset.
        When called from a child depth object, this function should provide
        all the kwargs needed to create a valid instance using the __init__.

        There are no universally required terms (yet)
        """
        df_vars = (
            dataset.variables
            if dataset is not None
            else get_dataset(filename).variables
        )
        df_vars = dict([(k.lower(), v) for k, v in df_vars.items()])
        init_args = {}
        dt = {}
        return init_args, dt

    @classmethod
    def _gen_varname(
        cls,
        filename=None,
        dataset=None,
        grid_dataset=None,
        names_list=None,
        std_names_list=None,
    ):
        """
        Function to find default variable names if they are not provided.

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
                if hasattr(var, "standard_name") or hasattr(var, "long_name"):
                    if var.name == n:
                        return n
        raise KeyError("Default names not found.")


class L_Depth(DepthBase):
    default_terms = ["depth_levels"]
    default_names = {"depth_levels": ["depth", "depth_levels", "Depth"]}
    cf_names = {"depth_levels": "depth"}

    def __init__(self, terms=None, **kwargs):
        super().__init__(**kwargs)
        self.terms = {}
        if terms is None:
            raise ValueError("Must provide terms for level depth coordinate")
        else:
            self.terms = terms
            for k, v in terms.items():
                setattr(self, k, v)

    @classmethod
    def _can_create_from_netCDF(
        cls,
        filename=None,
        dataset=None,
        grid_file=None,
        data_file=None,
    ):
        ds, dg = parse_filename_dataset_args(
            filename=filename,
            dataset=dataset,
            grid_file=grid_file,
            data_file=data_file,
        )

        return can_create_class(cls, ds, dg)

    @classmethod
    def from_netCDF(
        cls,
        filename=None,
        dataset=None,
        grid_file=None,
        data_file=None,
        name=None,
        topology=None,
        terms=None,
        **kwargs,
    ):
        df, dg = parse_filename_dataset_args(
            filename=filename,
            dataset=dataset,
            grid_file=grid_file,
            data_file=data_file,
        )
        nc_vars = search_netcdf_vars(cls, df, dg)
        if name is None:
            name = cls.__name__ + "_" + str(cls._instance_count)
        if terms is None:
            terms = {}
            for term, tvar in nc_vars.items():
                terms[term] = tvar[:]
        # 2023-02-21 set the depth of the top layer to zero
        surface_index = np.argmin(terms["depth_levels"])
        terms["depth_levels"][surface_index] = 0.0
        # 2023-02-21 set the depth of the top layer to zero

        return cls(name=name, terms=terms, **kwargs)

    @property
    def surface_index(self):
        return np.argmin(self.depth_levels)

    @property
    def bottom_index(self):
        return np.argmax(self.depth_levels)

    @property
    def num_levels(self):
        return len(self.depth_levels)

    @property
    def num_layers(self):
        return self.num_levels - 1

    def interpolation_alphas(
        self,
        points: np.ndarray | list | tuple,
        time: float | Time | None = None,
        data_shape: tuple[int, ...] | None = None,
        surface_boundary_condition: str | None = None,
        bottom_boundary_condition: str | None = None,
        extrapolate: bool = False,
        _hash: int | None = None,
        *args,
        **kwargs,
    ):
        """Calculates vertical layer indices and linear interpolation
        weights (alphas) for 3D coordinates.

        Determines which vertical layer interval each point resides in
        based on its depth component, accounting for both ascending
        (e.g., depths 0 to 100) and descending (e.g., elevations 100
        to 0) vertical coordinate tracking. Out-of-bounds points are
        handled based on custom surface and bottom boundary conditions.

        Args:
            points (array_like): An (N, 3) array of particle positions
                where the first column is longitude, the second
                latitude, and the third depth.
            time (float, datetime, or cftime.datetime, optional): Current
                timestamp. Can be a numeric offset matching dataset
                units (e.g. ROMS' time since...), a standard Python
                datetime, or a cftime datetime object for non-standard
                calendars. Defaults to None.
            data_shape (tuple, optional): The shape of the underlying
                variable grid (e.g., shape of u-/v-velocity).
                If None, the calculation defaults to the full 3D level
                coordinates. If the first dimension is 1 (indicating a
                single time step or a flat 2D grid), it falls back to
                surface-only evaluation. Defaults to None.
            surface_boundary_condition (str, optional): Boundary
                handling for points above the surface. Supported values:
                - 'mask': Masks out-of-bounds inputs in the outputs.
                - 'clamp': Pins out-of-bounds points to the surface.
                - 'extrapolate': Computes weights outside [0, 1].
                Defaults to self.default_surface_boundary_condition
            bottom_boundary_condition (str, optional): Boundary
                handling for points below the bottom floor. Supported values:
                - 'mask': Masks out-of-bounds inputs in the outputs.
                - 'clamp': Pins out-of-bounds points to the seabed.
                - 'extrapolate': Computes weights outside [0, 1].
                Defaults to self.default_bottom_boundary_condition.
            extrapolate (bool, optional): Whether to extrapolate
                values beyond boundaries. Defaults to False.
            _hash (int, optional): Cached grid hash identifier for
                performance optimization. Defaults to None.
            *args: Variable length argument list passed to fallback.
            **kwargs: Arbitrary keyword arguments passed to fallback.

        Returns:
            tuple: A pair of (indices, alphas) NumPy MaskedArrays:
                - indices (np.ma.MaskedArray): Integer array of shape
                  (N,) mapping each point to its lower bound depth
                  level index. Out-of-bounds indices are clamped to
                  boundary layers unless masked.
                - alphas (np.ma.MaskedArray): Float array of shape
                  (N,) containing the interpolation weight factor
                  between index and index+1. Values range between
                  [0.0, 1.0] within the grid layers.

        Raises:
            ValueError: If an unexpected calculation error occurs
                resulting in unmasked NaN alpha values.

        Examples:
            >>> # Querying a single particle's vertical metrics
            >>> u = gridded.Dataset(model).variables['u']
            >>> pt = [[-124.5, 45.2, 12.5]]  # lon, lat, depth
            >>> var_shape = (24, 40, 175, 120) # e.g., u.data_shape
            >>> idx, alpha = depth_info.interpolation_alphas(
            ...     points=pt,
            ...     data_shape=var_shape
            ... )
            >>> idx
            masked_array(data=[3], mask=[False], fill_value=999999)
            >>> alpha
            masked_array(data=[0.45], mask=[False], fill_value=1e+20)
        """
        points = np.asarray(points, dtype=np.float64)
        points = points.reshape(-1, 3)
        depths = points[:, 2]
        surface_boundary_condition = (
            self.default_surface_boundary_condition
            if surface_boundary_condition is None
            else surface_boundary_condition
        )
        bottom_boundary_condition = (
            self.default_bottom_boundary_condition
            if bottom_boundary_condition is None
            else bottom_boundary_condition
        )

        if (
            (data_shape is not None and data_shape[0] == 1)
            or self.num_levels == 1
        ):  # surface only
            return super(L_Depth, self).interpolation_alphas(
                points,
                time,
                data_shape,
                _hash=_hash,
                extrapolate=extrapolate,
                **kwargs,
            )

        # process remaining points that are 'above surface' or 'below ground'
        # L0 and L1 bound the entire vertical layer
        L0 = self.depth_levels[0]
        L1 = self.depth_levels[-1]
        ascending = L0 < L1

        # It is important to get the 'right' argument correct for np.digitize
        # See https://numpy.org/doc/stable/reference/generated/numpy.digitize.html
        idxs = (
            np.digitize(
                depths, self.depth_levels, right=False if ascending else True
            ) 
        ) - 1
        indices = np.ma.MaskedArray(data=idxs)
        alphas = np.ma.MaskedArray(
            data=np.full(len(points), np.nan, dtype=np.float64)
        )

        # Identify boundary conditions
        # set above surface and below seafloor alphas to allow future filtering
        if ascending:
            # 0, 1, 2, 3, 4, 5, 6
            above_surface = indices == -1
            above_alpha = 1
            below_bottom = indices == (len(self.depth_levels) - 1)
            below_alpha = 0
        else:
            # 6, 5, 4, 3, 2, 1, 0
            above_surface = indices == (len(self.depth_levels) - 1)
            above_alpha = 0
            below_bottom = indices == -1
            below_alpha = 1

        alphas[above_surface] = above_alpha
        alphas[below_bottom] = below_alpha

        # Apply boundary masks
        # set above surface and below seafloor mask
        if surface_boundary_condition == "mask":
            alphas.mask = np.logical_or(alphas.mask, above_surface)
            indices.mask[above_surface] = True
        if bottom_boundary_condition == "mask":
            alphas.mask = np.logical_or(alphas.mask, below_bottom)
            indices.mask[below_bottom] = True

        # RDM: CHECK...this is no longer in the main branch code
        # # Standardize boundary index values
        # indices[above_surface] = (
        #     0 if ascending else (len(self.depth_levels) - 2)
        # )
        # indices[below_bottom] = (
        #     (len(self.depth_levels) - 2) if ascending else 0
        # )

        # Interpolation within valid layers
        within_layer = np.isnan(alphas.data) & ~alphas.mask
        L0 = np.take(self.depth_levels, indices[within_layer])
        L1 = np.take(self.depth_levels, indices[within_layer] + 1)

        alphas[within_layer] = (depths[within_layer] - L0) / (L1 - L0)

        # Check for NaN values only in un-masked regions by zero
        # masked NaN values to zero
        if np.isnan(alphas.filled(0)).any():
            raise ValueError(
                "Some alphas are still unmasked and NaN. "
                "Please file a bug report"
            )

        return indices, alphas

    @classmethod
    def _find_required_depth_attrs(cls, filename, dataset=None, topology=None):

        # THESE ARE ACTUALLY ALL OPTIONAL. This should be migrated when optional
        # attributes are dealt with
        # Get superset attributes
        gf_vars = (
            dataset.variables
            if dataset is not None
            else get_dataset(filename).variables
        )
        gf_vars = dict([(k.lower(), v) for k, v in gf_vars.items()])
        init_args, gt = super()._find_required_depth_attrs(
            filename, dataset=dataset, topology=topology
        )

        return init_args, gt


class S_Depth(DepthBase):
    """
    Represents ocean s-coordinates as implemented by ROMS,
    It may or may not be useful for other systems.
    """

    _default_component_types = {
        "time": Time,
        "grid": Grid,
        "variable": None,
        "bathymetry": None,
        "zeta": None,
    }

    def __init__(
        self,
        name=None,
        time=None,
        grid=None,
        bathymetry=None,
        zeta=None,
        terms=None,
        vtransform=2,
        surface_boundary_condition="extrapolate",
        bottom_boundary_condition="mask",
        **kwargs,
    ):
        """
        :param name: Human readable name
        :type name: string

        :param time: time axis of the object
        :type time: gridded.time.Time or derivative

        :param grid: x/y grid representation
        :type grid: gridded.grids.GridBase or derivative

        :param bathymetry: variable object representing seafloor
        :type bathymetry: gridded.variable.Variable or derivative

        :param zeta: variable object representing free-surface
        :type zeta: gridded.variable.Variable or derivative

        :param terms: remaining terms in dictionary layout.
            e.g. ROMS: {'Cs_r': nc4.Dataset['Cs_r'][:],
                        'Cs_w': nc4.Dataset['Cs_w'][:],
                        's_rho': nc4.Dataset['s_rho'][:],
                        's_w': nc4.Dataset['s_w'][:],
                        'hc': nc4.Dataset['hc'][:],
        :type terms: dictionary of string key to numeric value.
                     See ``S_Depth.default_names``, sans bathymetry and zeta

        :param vtransform: S-coordinate transform type. 1 = Old, 2 = New
        :type vtransform: int (default 2)

        :param surface_boundary_condition: Determines how to handle points above the surface
        :type surface_boundary_condition: string ('extrapolate' or 'mask')

        :param bottom_boundary_condition: Determines how to handle points below the seafloor
        :type bottom_boundary_condition: string ('extrapolate' or 'mask')
        """

        super().__init__(**kwargs)
        self.name = name
        self.time = time
        self.grid = grid
        self.bathymetry = bathymetry  # Nodal bathymetry only.
        self.zeta = zeta
        self.terms = {}
        self.vtransform = vtransform
        if terms is None:
            raise ValueError("Must provide terms for sigma coordinate")
        else:
            self.terms = terms
            for k, v in terms.items():
                setattr(self, k, v)

        self.default_surface_boundary_condition = surface_boundary_condition
        self.default_bottom_boundary_condition = bottom_boundary_condition

    @classmethod
    def from_netCDF(
        cls,
        filename=None,
        grid_topology=None,
        name=None,
        time=None,
        time_origin=None,
        displacement=None,
        tz_offset=None,
        grid=None,
        dataset=None,
        data_file=None,
        grid_file=None,
        bathymetry=None,
        zeta=None,
        terms=None,
        vtransform=2,
        **kwargs,
    ):
        """
        :param filename: A string or ordered list of netCDF filename(s)
        :type filename: str or list[str]

        :param terms: Direct mapping of component name to netCDF variable name.
                     this if auto detection fails. Partial definition allowed.
                     Unspecified terms will use the value in `.default_names`.
                     ::
                        {'Cs_r': 'Cs_r',
                         'Cs_w': 'Cs_w',
                         's_rho': 's_rho'),
                         's_w': 's_w',
                         'bathymetry': 'h',
                         'hc': 'hc'),
                         'zeta': 'zeta')
                         }
        :type terms: dict

        :param name: Human-readable name for this object
        :type name: str

        :param time: Time dimension (for zeta)
        :type time: gridded.time.Time or subclass

        :param tz_offset: offset to compensate for time zone shifts
        :type tz_offset: `datetime.timedelta` or float or integer hours

        :param origin: shifts the time interval to begin at the time specified
        :type origin: `datetime.datetime`

        :param displacement: displacement to apply to the time data.
                             Allows shifting entire time interval into future
        :type displacement: `datetime.timedelta`

        :param grid: X-Y dmension (for bathymetry & zeta)
        :type grid: subclass of gridded.grids.GridBase
        """
        if cls == S_Depth:
            raise NotImplementedError(
                "S_Depth is not meant to be instantiated. Please use a "
                "subclass like ROMS_Depth or FVCOM_Depth"
            )

        Grid = cls._default_component_types["grid"]
        Time = cls._default_component_types["time"]
        Variable = cls._default_component_types["variable"]
        Bathymetry = cls._default_component_types["bathymetry"]
        Zeta = cls._default_component_types["zeta"]

        ds, dg = parse_filename_dataset_args(
            filename=filename,
            dataset=dataset,
            grid_file=grid_file,
            data_file=data_file,
        )

        if grid is None:
            grid = Grid.from_netCDF(dataset=dg, grid_topology=grid_topology)
        if name is None:
            name = cls.__name__ + "_" + str(cls._instance_count)
            cls._instance_count += 1

        # Do a comprehensive search for netCDF4 Variables all at once
        nc_vars = search_netcdf_vars(cls, ds, dg)

        if time is None:
            if zeta is None:
                zeta_var = nc_vars.get("zeta", None)
                if zeta_var is None:
                    warn = "Unable to locate zeta in data file"
                    if dg:
                        warnings.warn(warn + " or grid file.")
                    warn += " Generating constant (0) zeta."
                    warnings.warn(warn)
                    time = Time.constant_time()
                else:
                    time = Time.from_netCDF(
                        dataset=zeta_var._grp,
                        datavar=zeta_var,
                        origin=time_origin,
                        displacement=displacement,
                        tz_offset=tz_offset,
                    )
            elif isinstance(zeta, Variable) and zeta.time is not None:
                time = zeta.time
            else:
                time = Time.from_netCDF(
                    dataset=ds,
                    origin=time_origin,
                    displacement=displacement,
                    tz_offset=tz_offset,
                )

        if bathymetry is None:
            bathy_var = nc_vars.get("bathymetry", None)
            if bathy_var is None:
                err = "Unable to locate bathymetry in data file"
                if dg:
                    raise ValueError(err + " or grid file")
                raise ValueError(err)
            bathymetry = Bathymetry(
                data=bathy_var,
                grid=grid,
                name="bathymetry",
            )

        if zeta is None:
            zeta_var = nc_vars.get("zeta", None)
            if zeta_var is None:
                warn = "Unable to locate zeta in data file"
                if dg:
                    warnings.warn(warn + " or grid file.")
                warn += " Generating constant (0) zeta."
                warnings.warn(warn)
                zeta = Zeta.constant(0)
            else:
                zeta = Zeta(data=zeta_var, grid=grid, time=time, name="zeta")

        if terms is None:
            terms = {}
            for term, tvar in nc_vars.items():
                if term in ["bathymetry", "zeta"]:
                    # skip these because they're done separately...
                    continue
                terms[term] = tvar[:]
        if vtransform is None:
            vtransform = 2  # default for ROMS

        return cls(
            name=name,
            time=time,
            grid=grid,
            bathymetry=bathymetry,
            zeta=zeta,
            terms=terms,
            vtransform=vtransform,
            **kwargs,
        )

    @property
    def surface_index(self):
        raise NotImplementedError(
            "surface_index not implemented for S_Depth, required in subclasses"
        )

    @property
    def bottom_index(self):
        raise NotImplementedError(
            "bottom_index not implemented for S_Depth, required in subclasses"
        )

    @property
    def num_levels(self):
        raise NotImplementedError(
            "num_levels not implemented for S_Depth, required in subclasses"
        )

    @property
    def num_layers(self):
        raise NotImplementedError(
            "num_layers not implemented for S_Depth, required in subclasses"
        )

    @classmethod
    def _can_create_from_netCDF(
        cls,
        filename=None,
        dataset=None,
        grid_file=None,
        data_file=None,
    ):
        ds, dg = parse_filename_dataset_args(
            filename=filename,
            dataset=dataset,
            grid_file=grid_file,
            data_file=data_file,
        )

        found_vars = search_netcdf_vars(cls, ds, dg)
        if found_vars["zeta"] is None:
            found_vars.pop("zeta", None)
        return None not in found_vars.values()

    def __len__(self):
        return self.num_levels

    def get_transect(
        self, points, time, data_shape=None, _hash=None, **kwargs
    ):
        """
        :param points: array of points to interpolate to
        :type points: numpy array of shape (n, 3)

        :param time: time to interpolate to
        :type time: datetime.datetime

        :param data_shape: Shape of the variable to be interpolated.
        :type data_shape: tuple of int

        :return: numpy array of shape (n, data_shape[0]) of n depth level transects
        """
        raise NotImplementedError(
            "get_transect not implemented for S_Depth, required in subclasses"
        )

    def get_surface_depth(
        self, points, time, data_shape, _hash=None, **kwargs
    ):
        """
        :param points: array of points to interpolate to
        :type points: numpy array of shape (n, 3)

        :param time: time to interpolate to
        :type time: datetime.datetime

        :param data_shape: Shape of the variable to be interpolated.
        :type rho_or_w: tuple of int

        :return: numpy array of shape (n, 1) of n surface level values
        """
        raise NotImplementedError(
            "get_surface_depth not implemented for S_Depth, required in subclasses"
        )

    def interpolation_alphas(
        self,
        points: np.ndarray,
        time: float | Time | None,
        data_shape: tuple[int, ...],
        surface_boundary_condition: str | None = None,
        bottom_boundary_condition: str | None = None,
        _hash: int | None = None,
        extrapolate: bool = False,
        **kwargs,
    ) -> tuple[np.ma.MaskedArray, np.ma.MaskedArray]:
        """Calculates vertical layer indices and linear interpolation
        weights (alphas) for dynamic 3D transect profiles.

        Determines which vertical layer interval each point resides in
        by evaluating depth values against calculated water column
        transects. Handles coordinate tracking for both increasing
        (FVCOM) and decreasing (ROMS) grid structures. Out-of-bounds
        points are handled based on custom surface and bottom boundary
        conditions.

        If a depth is between coordinate N and N+1, the index will be N,
        with corresponding 1-alpha interpolation weight and alpha
        interpolation weight for N.  If a depth is exactly on a coordinate,
        the index will be N.

        Any points that are 'off grid' will be subject to the boundary
        conditions 'mask' or 'extrapolate'. 'mask' will mask the index
        and the alpha of the point.  'extrapolate' will set the index to
        the surface or bottom index, and the alpha to 0 or 1 depending on
        the orientation of the layers

        Args:
            points (np.ndarray): An (N, 3) array of particle positions
                where the first column is longitude, the second
                latitude, and the third depth.
            time (float, datetime, or cftime.datetime, optional):
                Current timestamp. Can be a numeric offset matching
                dataset units (e.g. ROMS' time since...), a standard
                Python datetime, or a cftime datetime object for
                non-standard calendars. Defaults to None.
            data_shape (tuple): The shape of the underlying variable
                grid (e.g., shape of u-/v-velocity). If None, the
                calculation defaults to the full 3D level coordinates.
                If the first dimension is 1 (indicating a single time
                step or a flat 2D grid), it falls back to surface-only
                evaluation.
            surface_boundary_condition (str, optional): Boundary
                handling for points above the surface. Supported
                values:
                - 'mask': Masks out-of-bounds inputs in the outputs.
                - 'clamp': Pins out-of-bounds points to the surface.
                - 'extrapolate': Computes weights outside [0, 1].
                Defaults to self.default_surface_boundary_condition.
            bottom_boundary_condition (str, optional): Boundary
                handling for points below the bottom floor. Supported
                values:
                - 'mask': Masks out-of-bounds inputs in the outputs.
                - 'clamp': Pins out-of-bounds points to the seabed.
                - 'extrapolate': Computes weights outside [0, 1].
                Defaults to self.default_bottom_boundary_condition.
            extrapolate (bool, optional): Whether to extrapolate
                values beyond boundaries. Defaults to False.
            _hash (int, optional): Cached grid hash identifier for
                performance optimization. Defaults to None.
            **kwargs: Arbitrary keyword arguments passed to fallback.

        Returns:
            tuple: A pair of (indices, alphas) NumPy MaskedArrays:
                - indices (np.ma.MaskedArray): Integer array of shape
                  (N,) containing the lower depth level index of the
                  interpolation, corresponding to an alpha of
                  alpha = 1 - alpha. Out-of-bounds indices are handled
                  according to boundary conditions.  The alpha for
                  indices[0] + 1 level is just alpha.
                - alphas (np.ma.MaskedArray): Float array of shape
                  (N,) containing the interpolation weight factor
                  for the upper "indices[0] + 1" level. Values range
                  between [0.0, 1.0] within the interpolation layers.

        Raises:
            ValueError: If the data_shape does not match level or
                layer dimensions, or if calculated alphas contain
                unmasked NaNs or values outside [0, 1].

        Examples:
            >>> # Querying a single particle's vertical metrics
            >>> u = gridded.Dataset(model).variables['u']
            >>> pt = [[-124.5, 45.2, 12.5]]  # lon, lat, depth
            >>> var_shape = (24, 40, 175, 120) # e.g., u.data_shape
            >>> idx, alpha = depth_info.interpolation_alphas(
            ...     points=pt,
            ...     data_shape=var_shape
            ... )
            >>> idx
            masked_array(data=[3], mask=[False], fill_value=999999)
            >>> alpha
            masked_array(data=[0.45], mask=[False], fill_value=1e+20)
        """
        depths = points[:, 2]
        surface_boundary_condition = (
            self.default_surface_boundary_condition
            if surface_boundary_condition is None
            else surface_boundary_condition
        )
        bottom_boundary_condition = (
            self.default_bottom_boundary_condition
            if bottom_boundary_condition is None
            else bottom_boundary_condition
        )
        surface_index = self.surface_index
        bottom_index = self.bottom_index

        # Surface-only fallback evaluation
        if (
            data_shape is not None and data_shape[0] == 1
        ) or self.num_levels == 1:
            return super().interpolation_alphas(
                points,
                time,
                data_shape,
                _hash=_hash,
                extrapolate=extrapolate,
                **kwargs,
            )

        if (
            data_shape[0] != self.num_levels
            and data_shape[0] != self.num_layers
        ):
            raise ValueError(
                "Cannot get depth interpolation alphas for data shape "
                "specified; does not fit r or w depth axis"
            )

        transects = self.get_transect(
            points,
            time,
            data_shape=data_shape,
            _hash=_hash,
            extrapolate=extrapolate,
        )

        # use np.digitize to bin the depths into the layers.
        # https://numpy.org/doc/stable/reference/generated/numpy.digitize.html
        # bins[i-1] <= x < bins[i] for FVCOM (right=False, increasing order)
        # bins[i-1] > x >= bins[i] for ROMS (right=False, decreasing order)
        # surface level will be 'within bounds' and the seafloor level will 
        # NOT be
        vf = np.vectorize(
            np.digitize, signature="(),(n)->()", excluded=["right"]
        )
        digitized_idxs = vf(depths, transects, right=False) - 1

        # Re-apply horizontal boundary tracking via transect mask
        indices = np.ma.MaskedArray(digitized_idxs, mask=transects.mask[:, 0])

        alphas = np.ma.MaskedArray(
            data=np.full(len(points), np.nan, dtype=np.float64),
            mask=transects.mask[:, 0],
        )

        indices, alphas, oob_mask = self._apply_boundary_conditions(
            indices,
            alphas,
            surface_index,
            bottom_index,
            surface_boundary_condition,
            bottom_boundary_condition,
        )

        idx = np.clip(
            np.ma.filled(indices, 0), 0, transects.shape[1] - 2
        )[:, np.newaxis]
        L0 = np.take_along_axis(transects, idx, axis=1).squeeze(axis=1)
        L1 = np.take_along_axis(transects, idx + 1, axis=1).squeeze(axis=1)

        within_layer = np.isnan(alphas.data) & ~alphas.mask

        alphas[within_layer] = (depths[within_layer] - L0[within_layer]) / (
            L1[within_layer] - L0[within_layer]
        )

        # Final pass validation checking actual data fields
        if np.isnan(alphas.filled(0)).any():
            raise ValueError(
                "Some alphas are still unmasked and NaN. "
                "Please file a bug report"
            )
        if (alphas.filled(0) < 0).any() or (alphas.filled(0) > 1).any():
            print(
                f"depths.py[921], depth - lower level depth <0 or >1: {alphas}"
            )
            if depths[within_layer] < L1:  # between rho-depth and surface
                indices, alphas = indices, alphas * 0.0
            else:
                raise ValueError(
                    "Some alphas are outside the range [0, 1]. "
                    "Please file a bug report"
                )

        return indices, alphas

    def get_s_coord(
        self, points, time, data_shape=None, _hash=None, **kwargs
    ):
        raise NotImplementedError(
            "get_s_coord not implemented for S_Depth, required in subclasses"
        )

    def get_transect(
        self, points, time, data_shape=None, _hash=None, **kwargs
    ):
        z = self.zeta.at(points, time, unmask=False, _hash=_hash, **kwargs)
        return (
            self.get_s_coord(
                points, time, data_shape=data_shape, _hash=_hash, **kwargs
            )
            + z
        )

    def _apply_boundary_conditions(
        self,
        indices,
        alphas,
        surface_index=None,
        bottom_index=None,
        surface_boundary_condition=None,
        bottom_boundary_condition=None,
    ):
        surface_index = (
            self.surface_index if surface_index is None else surface_index
        )
        bottom_index = (
            self.bottom_index if bottom_index is None else bottom_index
        )
        surface_boundary_condition = (
            self.default_surface_boundary_condition
            if surface_boundary_condition is None
            else surface_boundary_condition
        )
        bottom_boundary_condition = (
            self.default_bottom_boundary_condition
            if bottom_boundary_condition is None
            else bottom_boundary_condition
        )

        if surface_index == 0:
            above_surf_mask = indices < surface_index
            below_bottom_mask = indices >= bottom_index
            alphas[above_surf_mask] = 1
            alphas[below_bottom_mask] = 0
        else:
            above_surf_mask = indices >= surface_index
            below_bottom_mask = indices < bottom_index
            alphas[above_surf_mask] = 0
            alphas[below_bottom_mask] = 1
        oob_mask = np.logical_or(above_surf_mask, below_bottom_mask)
        indices.mask = np.logical_or(indices.mask, oob_mask)
        alphas.mask = np.logical_or(alphas.mask, oob_mask)

        if surface_boundary_condition == "extrapolate":
            indices.mask[above_surf_mask] = False
            alphas.mask[above_surf_mask] = False
        if bottom_boundary_condition == "extrapolate":
            indices.mask[below_bottom_mask] = False
            alphas.mask[below_bottom_mask] = False

        return indices, alphas, oob_mask


class ROMS_Depth(S_Depth):
    _instance_count = 0
    default_names = {
        "Cs_r": ["Cs_r"],
        "Cs_w": ["Cs_w"],
        "s_rho": ["s_rho"],
        "s_w": ["s_w"],
        "hc": ["hc"],
        "bathymetry": ["h"],
        "zeta": ["zeta"],
    }

    cf_names = {
        "Cs_r": ["S-coordinate stretching curves at RHO-points"],
        "Cs_w": ["S-coordinate stretching curves at W-points"],
        "s_rho": ["S-coordinate at RHO-points"],
        "s_w": ["S-coordinate at W-points"],
        "hc": ["S-coordinate parameter, critical depth"],
        "bathymetry": [
            "bathymetry at RHO-points",
            "Final bathymetry at RHO-points",
        ],
        "zeta": ["free-surface"],
    }

    @property
    def surface_index(self):
        return np.argmax(self.s_w)

    @property
    def bottom_index(self):
        return np.argmin(self.s_w)

    @property
    def num_levels(self):
        return len(self.s_w)

    @property
    def num_layers(self):
        return len(self.s_rho)

    def get_transect(
        self, points, time, data_shape=None, _hash=None, **kwargs
    ):
        if data_shape is None:
            data_shape = (self.num_levels,)

        s_c = self.s_rho if data_shape[0] == self.num_layers else self.s_w
        C_s = self.Cs_r if data_shape[0] == self.num_layers else self.Cs_w
        h = self.bathymetry.at(
            points, time, unmask=False, _hash=_hash, **kwargs
        )
        zeta = self.zeta.at(points, time, unmask=False, _hash=_hash, **kwargs)
        hc = self.hc
        hCs = h * C_s[np.newaxis, :]
        # Fixed transect such that depth is from air-sea interface
        # rather than geoid
        if self.vtransform == 1:
            S = (hc * s_c) + hCs - (hc * C_s)[np.newaxis, :]
            s_coord = -(S + zeta * (1 + S / h)) # RDM -(S + zeta * (1 + S / h))
        elif self.vtransform == 2:
            S = ((hc * s_c) + hCs) / (hc + h)
            s_coord = -(zeta + h) * S #RDM -(zeta + (zeta + h) * S)
        return s_coord


class FVCOM_Depth(S_Depth):
    _instance_count = 0
    default_names = {
        "siglay": ["siglay"],
        "siglay_center": ["siglay_center"],
        "siglev": ["siglev"],
        "siglev_center": ["siglev_center"],
        "bathymetry": ["h"],
        "h_center": ["h_center"],
        "zeta": ["zeta"],
    }

    cf_names = {
        "siglay": ["ocean_sigma/general_coordinate"],
        "siglay_center": ["ocean_sigma/general_coordinate"],
        "siglev": ["ocean_sigma/general_coordinate"],
        "siglev_center": ["ocean_sigma/general_coordinate"],
        "bathymetry": ["sea_floor_depth_below_geoid"],
        "h_center": ["sea_floor_depth_below_geoid"],
        "zeta": ["sea_surface_height_above_geoid"],
    }

    @property
    def surface_index(self):
        return np.argmax(self.siglev[:, 0])

    @property
    def bottom_index(self):
        return np.argmin(self.siglev[:, 0])

    @property
    def num_levels(self):
        return len(self.siglev)

    @property
    def num_layers(self):
        return len(self.siglay)

    def get_s_coord(
        self, points, time, data_shape=None, _hash=None, **kwargs
    ):
        sigvar = None
        if data_shape is None:
            sigvar = self.siglev[:].T
        elif data_shape[0] == self.num_layers:
            sigvar = self.siglay[:].T
        else:
            sigvar = self.siglev[:].T
        sigma = self.grid.interpolate_var_to_points(
            points[:, 0:2], sigvar, location="node"
        )

        bathy = self.bathymetry.at(
            points, time, unmask=False, _hash=_hash, **kwargs
        )
        zeta = self.zeta.at(points, time, unmask=False, _hash=_hash, **kwargs)

        s_coord = -(zeta + (zeta + bathy) * sigma)
        return s_coord


class Depth:
    """
    Factory class that generates depth objects.

    Also handles common loading and parsing operations
    """

    ld_types = [L_Depth]
    sd_types = [ROMS_Depth, FVCOM_Depth]
    surf_types = [DepthBase]

    def __init__(self):
        raise NotImplementedError(
            "Depth is not meant to be instantiated. Please use the "
            "'from_netCDF' or 'surface_only' function"
        )

    @staticmethod
    def surface_only(surface_index=-1, **kwargs):
        return DepthBase(surface_index=surface_index, **kwargs)

    @classmethod
    def from_netCDF(
        cls,
        filename=None,
        dataset=None,
        data_file=None,
        grid_file=None,
        **kwargs,
    ):
        ds, dg = parse_filename_dataset_args(
            filename=filename,
            dataset=dataset,
            data_file=data_file,
            grid_file=grid_file,
        )

        typs = cls.sd_types + cls.ld_types
        available_to_create = [
            typ._can_create_from_netCDF(grid_file=dg, data_file=ds)
            for typ in typs
        ]
        if not any(available_to_create):
            warnings.warn(
                "Unable to automatically determine depth system so "
                "reverting to surface-only mode. Manually check the "
                "(depth_object).surface_index attribute and set it "
                "to the appropriate array index for your model data",
                RuntimeWarning,
            )
            return cls.surf_types[0].from_netCDF(
                data_file=ds, grid_file=dg, **kwargs
            )
        else:
            typ = typs[np.argmax(available_to_create)]
            if sum(available_to_create) > 1:
                warnings.warn(
                    "Multiple depth systems detected. "
                    f"Using the first one found: {typ!r}",
                    RuntimeWarning,
                )

            return typ.from_netCDF(
                filename=filename,
                dataset=dataset,
                data_file=data_file,
                grid_file=grid_file,
                **kwargs,
            )