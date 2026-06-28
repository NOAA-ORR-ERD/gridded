import warnings
import os

import numpy as np
from gridded.time import Time
from gridded.grids import Grid
from gridded.utilities import (get_dataset,
                               search_dataset_for_any_long_name,
                               search_dataset_for_variables_by_longname,
                               search_dataset_for_variables_by_varname,
                               merge_var_search_dicts,
                               search_netcdf_vars,
                               can_create_class,
                               parse_filename_dataset_args
                               )

class DepthBase(object):

    _instance_count = 0
    _default_component_types = {'time': Time,
                                'grid': Grid,
                                'variable': None}
    #'variable' is None here to avoid import issues. It is set in the __init__.py

    def __init__(self,
                 name=None,
                 surface_index=-1,
                 bottom_index=0,
                 default_surface_boundary_condition='extrapolate',
                 default_bottom_boundary_conditon='mask',
                 **kwargs):
        '''
        :param surface_index: array index of 'highest' level (closest to sea level)

        :param bottom_index: array index of 'lowest' level (closest to seafloor)
        '''
        self.name = name
        self._surface_index = surface_index
        self._bottom_index = bottom_index
        self.default_surface_boundary_condition = default_surface_boundary_condition
        self.default_bottom_boundary_condition = default_bottom_boundary_conditon

    @classmethod
    def _can_create_from_netCDF(cls,
                                filename=None,
                                dataset=None,
                                grid_file=None,
                                data_file=None,):
        return True

    @classmethod
    def from_netCDF(cls,
                    surface_index=-1,
                    **kwargs):
        return cls(surface_index,
                   **kwargs)

    @property
    def surface_index(self):
        #Subclasses are REQUIRED to implement this property in the manner appropriate
        #for the system being represented
        return self._surface_index

    @property
    def bottom_index(self):
        #Subclasses are REQUIRED to implement this property in the manner appropriate
        #for the system being represented
        return self._bottom_index

    def interpolation_alphas(self,
                             points,
                             time,
                             data_shape,
                             extrapolate=False,
                             _hash=None,
                             **kwargs,):
        '''
        Returns the weights (alphas) required for interpolation

        The base class implementation only supports surface values:
        it returns the surface index and 0.0 for the alpha

        This is the expected outcome for the case where all points
        are on the surface.
        '''
        sz = len(points)
        indices = np.ma.MaskedArray(data=np.ones((sz, ), dtype=np.int64) * self.surface_index,
                                    mask=np.zeros((sz, ), dtype=bool))
        alphas = np.ma.MaskedArray(data=np.zeros((sz, ), dtype=np.float64),
                                   mask=np.zeros((sz, ), dtype=bool))
        return indices, alphas

    @classmethod
    def _find_required_depth_attrs(cls, filename, dataset=None, depth_topology=None):
        '''
        This function is the top level 'search for attributes' function. If there are any
        common attributes to all potential depth types, they will be sought here.

        This function returns a dict, which maps an attribute name to a netCDF4
        Variable or numpy array object extracted from the dataset. When called from
        a child depth object, this function should provide all the kwargs needed to
        create a valid instance using the __init__.

        There are no universally required terms (yet)
        '''
        df_vars = dataset.variables if dataset is not None else get_dataset(filename).variables
        df_vars = dict([(k.lower(), v) for k, v in df_vars.items()] )
        init_args = {}
        dt = {}
        return init_args, dt

    @classmethod
    def _gen_varname(cls,
                     filename=None,
                     dataset=None,
                     grid_dataset=None,
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
                if hasattr(var, 'standard_name') or hasattr(var, 'long_name'):
                    if var.name == n:
                        return n
        raise KeyError("Default names not found.")


class L_Depth(DepthBase):

    default_terms = [('depth_levels')]
    default_names = {'depth_levels': ['depth','depth_levels', 'Depth']}
    cf_names = {'depth_levels': 'depth'}

    def __init__(self,
                 terms=None,
                 **kwargs):
        super(L_Depth, self).__init__(**kwargs)
        self.terms = {}
        if terms is None:
            raise ValueError('Must provide terms for level depth coordinate')
        else:
            self.terms = terms
            for k, v in terms.items():
                setattr(self, k, v)

    @classmethod
    def _can_create_from_netCDF(cls,
                                filename=None,
                                dataset=None,
                                grid_file=None,
                                data_file=None,):
            ds, dg = parse_filename_dataset_args(filename=filename,
                                                 dataset=dataset,
                                                 grid_file=grid_file,
                                                 data_file=data_file)

            return can_create_class(cls, ds, dg)
            
    @classmethod
    def from_netCDF(cls,
                    filename=None,
                    dataset=None,
                    grid_file=None,
                    data_file=None,
                    name=None,
                    topology=None,
                    terms=None,
                    **kwargs
                    ):
        df, dg = parse_filename_dataset_args(filename=filename,
                                             dataset=dataset,
                                             grid_file=grid_file,
                                             data_file=data_file)
        nc_vars = search_netcdf_vars(cls, df, dg)
        if name is None:
            name = cls.__name__ + '_' + str(cls._instance_count)
        if terms is None:
            terms = {}
            for term, tvar in nc_vars.items():
                terms[term] = tvar[:]
        # 2023-02-21 set the depth of the top layer to zero
        surface_index = np.argmin(terms['depth_levels'])
        terms['depth_levels'][surface_index] = 0.0
        # 2023-02-21 set the depth of the top layer to zero

        return cls(name=name,
                   terms=terms,
                   **kwargs)

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
        """ Calculates vertical layer indices and linear interpolation 
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
                units (e.g. ROMS' time since...), a standard Python datetime, 
                or a cftime datetime object for non-standard calendars. 
                Defaults to None.
            data_shape (tuple, optional): The shape of the underlying 
                variable grid (e.g., shape of u-/v-velocity). 
                If None, the calculation defaults to the full 3D level 
                coordinates. If the first dimension is 1 (indicating a 
                single time step or a flat 2D grid), it falls back to 
                surface-only evaluation. Defaults to None.
            surface_boundary_condition (str, optional): Boundary 
                handling for points above the surface. 
                Supported values:
                - 'mask': Masks out-of-bounds inputs in the outputs.
                - 'clamp': Pins out-of-bounds points to the surface.
                - 'extrapolate': Computes weights outside [0, 1].
                Defaults to self.default_surface_boundary_condition
            bottom_boundary_condition (str, optional): Boundary 
                handling for points below the bottom floor. 
                Supported values:
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
        surface_boundary_condition = self.default_surface_boundary_condition if surface_boundary_condition is None else surface_boundary_condition
        bottom_boundary_condition = self.default_bottom_boundary_condition if bottom_boundary_condition is None else bottom_boundary_condition
        
        if data_shape is not None and data_shape[0] == 1 or self.num_levels == 1: #surface only
            return super(L_Depth, self).interpolation_alphas(points, time, data_shape, _hash=_hash, extrapolate=extrapolate, **kwargs)
        
        # process remaining points that are 'above the surface' or 'below the ground'
        # L0 and L1 bound the entire vertical layer
        # It is important to get the 'right' argument correct for np.digitize
        # See https://numpy.org/doc/stable/reference/generated/numpy.digitize.html
        L0 = self.depth_levels[0]
        L1 = self.depth_levels[-1]
        ascending = L0 < L1

        # Digitization
        idxs = np.digitize(
            depths, self.depth_levels, right=False if ascending else True
        ) - 1
        indices = np.ma.MaskedArray(data=idxs)
        alphas = np.ma.MaskedArray(