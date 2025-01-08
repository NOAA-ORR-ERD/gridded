
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

    # hack so that older code can access _def_count
    @property
    def _def_count(self):
        return self._instance_count
    @_def_count.setter
    def _def_count(self, count):
        self.__class__._instance_count = count

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

    def interpolation_alphas(self,
                             points, 
                             time = None,
                             data_shape=None,
                             surface_boundary_condition=None,
                             bottom_boundary_condition=None,
                             extrapolate=False,
                             _hash=None,
                             *args,
                             **kwargs):
        '''
        Returns a pair of values.

        The 1st value is an array of the depth indices of all the particles.

        The 2nd value is an array of the interpolation alphas for the particles
        between their depth index and depth_index+1. If both values are None,
        then all particles are on the surface layer.
        '''
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
        idxs = np.digitize(depths, self.depth_levels, right=False if L0 < L1 else True) - 1
        indices = np.ma.MaskedArray(data=idxs, mask=np.zeros((len(idxs)), dtype=bool))

        alphas = np.ma.MaskedArray(data=np.empty((len(points)), dtype=np.float64) * np.nan, mask=np.zeros((len(points)), dtype=bool))

        # set above surface and below seafloor alphas to allow future filtering

        if L0 < L1:
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
        
        #set above surface and below seafloor mask
        if surface_boundary_condition == 'mask':
            alphas.mask = np.logical_or(alphas.mask, above_surface)
            indices.mask[above_surface] = True
        alphas[indices == -1] = 1
        if bottom_boundary_condition == 'mask':
            alphas.mask = np.logical_or(alphas.mask, below_bottom)
            indices.mask[below_bottom] = True
        
        within_layer = np.isnan(alphas)
        
        L0 = np.take(self.depth_levels, indices[within_layer])
        L1 = np.take(self.depth_levels, indices[within_layer] + 1)
        
        alphas[within_layer] = (depths[within_layer] - L0) / (
            L1 - L0)
        
        if any(np.isnan(alphas)):
            raise ValueError('Some alphas are still unmasked and NaN. Please file a bug report')
        
        return indices, alphas

    @classmethod
    def _find_required_depth_attrs(cls, filename, dataset=None, topology=None):

        # THESE ARE ACTUALLY ALL OPTIONAL. This should be migrated when optional
        #   attributes are dealt with
        # Get superset attributes
        gf_vars = dataset.variables if dataset is not None else get_dataset(filename).variables
        gf_vars = dict([(k.lower(), v) for k, v in gf_vars.items()] )
        init_args, gt = super(L_Depth, cls)._find_required_depth_attrs(filename,
                                                                       dataset=dataset,
                                                                       topology=topology)

        return init_args, gt

class S_Depth(DepthBase):
    '''
    Represents the ocean s-coordinates as implemented by ROMS,
    It may or may not be useful for other systems.
    '''
    _default_component_types = {'time': Time,
                                'grid': Grid,
                                'variable': None,
                                'bathymetry': None,
                                'zeta': None,}

    def __init__(self,
                 name=None,
                 time=None,
                 grid=None,
                 bathymetry=None,
                 zeta=None,
                 terms=None,
                 vtransform=2,
                 surface_boundary_condition='extrapolate',
                 bottom_boundary_condition='mask',
                 **kwargs):
        '''
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

        :param terms: remaining terms in dictionary layout
        :type terms: dictionary of string key to numeric value
        See S_Depth.default_names, sans bathymetry and zeta

        :param vtransform: S-coordinate transform type. 1 = Old, 2 = New
        :type vtransform: int (default 2)

        :param surface_boundary_condition: Determines how to handle points above the surface
        :type surface_boundary_condition: string ('extrapolate' or 'mask')

        :param bottom_boundary_condition: Determines how to handle points below the seafloor
        :type bottom_boundary_condition: string ('extrapolate' or 'mask')
        '''

        super(S_Depth, self).__init__(**kwargs)
        self.name = name
        self.time = time
        self.grid = grid
        self.bathymetry = bathymetry # Nodal bathymetry only.
        self.zeta = zeta
        self.terms = {}
        self.vtransform = vtransform
        if terms is None:
            raise ValueError('Must provide terms for sigma coordinate')
        else:
            self.terms=terms
            for k, v in terms.items():
                setattr(self, k, v)

        # self.rho_coordinates = self.compute_coordinates('rho')
        # self.w_coordinates = self.compute_coordinates('w')
        self.default_surface_boundary_condition = surface_boundary_condition
        self.default_bottom_boundary_condition = bottom_boundary_condition
    @classmethod
    def from_netCDF(cls,
                    filename=None,
                    varnames=None,
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
                    # load_all=False,
                    bathymetry=None,
                    zeta=None,
                    terms=None,
                    vtransform=2,
                    **kwargs
                    ):
        '''
        :param filename: A string or ordered list of string of netCDF filename(s)
        :type filename: str or list[str]

        :param varnames: Direct mapping of component name to netCDF variable name. Use
                         this if auto detection fails. Partial definition is allowable.
                         Unspecified terms will use the value in `.default_names`::
                           {'Cs_r': 'Cs_r',
                            'Cs_w': Cs_w',
                            's_rho': 's_rho'),
                            's_w': 's_w',
                            'bathymetry': 'h',
                            'hc': 'hc'),
                            'zeta': 'zeta')
                            }
        :type varnames: dict

        :param name: Human-readable name for this object
        :type name: str

        :param time: Time dimension (for zeta)
        :type time: gridded.time.Time or subclass

        :param tz_offset: offset to compensate for time zone shifts
        :type tz_offset: `datetime.timedelta` or float or integer hours

        :param origin: shifts the time interval to begin at the time specified
        :type origin: `datetime.datetime`

        :param displacement: displacement to apply to the time data.
               Allows shifting entire time interval into future or past
        :type displacement: `datetime.timedelta`
        
        :param grid: X-Y dmension (for bathymetry & zeta)
        :type grid: subclass of gridded.grids.GridBase
        '''
        Grid = cls._default_component_types['grid']
        Time = cls._default_component_types['time']
        Variable = cls._default_component_types['variable']
        Bathymetry = cls._default_component_types['bathymetry']
        Zeta = cls._default_component_types['zeta']

        ds, dg = parse_filename_dataset_args(filename=filename,
                                             dataset=dataset,
                                             grid_file=grid_file,
                                             data_file=data_file)

        if grid is None:
            grid = Grid.from_netCDF(dataset=dg,
                                    grid_topology=grid_topology)
        if name is None:
            name = cls.__name__ + '_' + str(cls._instance_count)
            cls._instance_count += 1
        
        # Do a comprehensive search for netCDF4 Variables all at once
        nc_vars = search_netcdf_vars(cls, ds, dg)
        
        if time is None:
            zeta_var = nc_vars.get('zeta', None)
            if zeta_var is None:
                warn = 'Unable to locate zeta in data file'
                if dg:
                    warnings.warn(warn + ' or grid file.')
                warn += ' Generating constant (0) zeta.'
                warnings.warn(warn)
                time = Time.constant_time()
            else:
                time = Time.from_netCDF(
                    datavar=zeta_var,
                    filename=data_file,
                    origin=time_origin,
                    displacement=displacement,
                    tz_offset=tz_offset
                )
                                        
        if bathymetry is None:
            bathy_var = nc_vars.get('bathymetry', None)
            if bathy_var is None:
                err = 'Unable to locate bathymetry in data file'
                if dg:
                    raise ValueError(err + ' or grid file')
                raise ValueError(err)
            bathymetry = Bathymetry(data=bathy_var,
                                    grid=grid,
                                    name='bathymetry',
                                    )

        if zeta is None:
            zeta_var = nc_vars.get('zeta', None)
            if zeta_var is None:
                warn = 'Unable to locate zeta in data file'
                if dg:
                    warnings.warn(warn + ' or grid file.')
                warn += ' Generating constant (0) zeta.'
                warnings.warn(warn)
                zeta = Zeta.constant(0)
            else:
                zeta = Zeta(data=zeta_var,
                            grid=grid,
                            time=time,
                            name='zeta')

        if terms is None:
            terms = {}
            for term, tvar in nc_vars.items():
                if term in ['bathymetry', 'zeta']:
                    # skip these because they're done separately...
                    continue
                terms[term] = tvar[:]
        if vtransform is None:
            vtransform = 2  #default for ROMS
            #  messing about trying to detect this.

        return cls(name=name,
                   time=time,
                   grid=grid,
                   bathymetry=bathymetry,
                   zeta=zeta,
                   terms=terms,
                   vtransform=vtransform,
                   **kwargs)

    @property
    def surface_index(self):
        raise NotImplementedError('surface_index not implemented for S_Depth, required in subclasses')

    @property
    def bottom_index(self):
        raise NotImplementedError('bottom_index not implemented for S_Depth, required in subclasses')

    @property
    def num_levels(self):
        raise NotImplementedError('num_levels not implemented for S_Depth, required in subclasses')

    @property
    def num_layers(self):
        raise NotImplementedError('num_layers not implemented for S_Depth, required in subclasses')

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

        found_vars = search_netcdf_vars(cls, ds, dg)
        #necessary to support optional zeta when called from Depth.from_netCDF
        #this is a hack that circumvents the 'can_create_class' function
        #what we really need is a way to specify that a sought attriubute is optional
        #in the 'schema' (default_names, cf_names, etc)
        if found_vars['zeta'] is None:
            found_vars.pop('zeta', None)
        # all variables must be found (no None values)
        return not (None in found_vars.values())

    def __len__(self):
        return self.num_levels
        
    def get_transect(self, points, time, data_shape=None, _hash=None, **kwargs):
        '''
        :param points: array of points to interpolate to
        :type points: numpy array of shape (n, 3)
        
        :param time: time to interpolate to
        :type time: datetime.datetime
        
        :param data_shape: Shape of the variable to be interpolated. The first dimension is expected to be depth
        :type data_shape: tuple of int
        
        :return: numpy array of shape (n, data_shape[0]) of n depth level transects
        '''
        raise NotImplementedError('get_transect not implemented for S_Depth, required in subclasses')

    def get_surface_depth(self, points, time, data_shape, _hash=None, **kwargs):
        '''
        :param points: array of points to interpolate to
        :type points: numpy array of shape (n, 3)

        :param time: time to interpolate to
        :type time: datetime.datetime

        :param data_shape: Shape of the variable to be interpolated. The first dimension is expected to be depth
        :type rho_or_w: tuple of int

        :return: numpy array of shape (n, 1) of n surface level values
        '''
        raise NotImplementedError('get_surface_depth not implemented for S_Depth, required in subclasses')


    def interpolation_alphas(self,
                             points,
                             time,
                             data_shape,
                             surface_boundary_condition=None,
                             bottom_boundary_condition=None,
                             _hash=None,
                             extrapolate=False,
                             **kwargs):
        '''
        Returns a pair of arrays. The 1st array is of the depth indices
        of all the points. The 2nd value is an array of the interpolation
        alphas for the points between their depth index and depth_index+1.

        If a depth is between coordinate N and N+1, the index will be N.
        If a depth is exactly on a coordinate, the index will be N.

        Any points that are 'off grid' will be subject to the boundary conditions
        'mask' or 'extrapolate'. 'mask' will mask the index and the alpha of the point.
        'extrapolate' will set the index to the surface or bottom index, and the alpha to
        0 or 1 depending on the orientation of the layers
        '''
        depths = points[:,2]
        surface_boundary_condition = self.default_surface_boundary_condition if surface_boundary_condition is None else surface_boundary_condition
        bottom_boundary_condition = self.default_bottom_boundary_condition if bottom_boundary_condition is None else bottom_boundary_condition

        surface_index = self.surface_index
        bottom_index = self.bottom_index

        if data_shape is not None and data_shape[0] == 1 or self.num_levels == 1: #surface only
            return super(S_Depth, self).interpolation_alphas(points, time, data_shape, _hash=_hash, extrapolate=extrapolate, **kwargs)

        if data_shape[0] != self.num_levels and data_shape[0] != self.num_layers:
            raise ValueError('Cannot get depth interpolation alphas for data shape specified; does not fit r or w depth axis')
        #if data_shape[0] == self.num_layers:
        #    raise NotImplementedError('Interpolation of data on depth layers not supported yet')

        transects = self.get_transect(points, time, data_shape=data_shape, _hash=_hash, extrapolate=extrapolate)

        indices = np.ma.MaskedArray(data=-np.ones((len(points)), dtype=np.int64) * 1000, mask=np.zeros((len(points)), dtype=bool))
        alphas = np.ma.MaskedArray(data=np.empty((len(points)), dtype=np.float64) * np.nan, mask=np.zeros((len(points)), dtype=bool))

        #use np.digitize to bin the depths into the layers.
        #https://numpy.org/doc/stable/reference/generated/numpy.digitize.html
        #bins[i-1] <= x < bins[i] should be satisfied for FVCOM (right=False, increasing order)
        #bins[i-1] > x >= bins[i] should be satisfied for ROMS (right=False, decreasing order)
        #this means the surface level will be 'within bounds' and the seafloor level will NOT be
        vf = np.vectorize(np.digitize, signature='(),(n)->()', excluded=['right'])
        indices = vf(depths, transects, right=False) - 1
        
        # transect mask is True where the point is outside the grid horizontally
        # so it must be reapplied
        indices = np.ma.array(indices, mask=transects.mask[:,0])
        alphas.mask = transects.mask[:,0]
        indices, alphas, oob_mask = self._apply_boundary_conditions(indices,
                                                                    alphas,
                                                                    surface_index,
                                                                    bottom_index,
                                                                    surface_boundary_condition,
                                                                    bottom_boundary_condition)
        
        # compute the remaining alphas, which should be for points within the depth interval
        L0 = np.take(transects, indices)
        L1 = np.take(transects, indices + 1)
        within_layer = np.isnan(alphas) #remaining alphas would still have nan at this point
        alphas[within_layer] = (depths[within_layer] - L0[within_layer]) / (L1[within_layer] - L0[within_layer])
        
        if any(np.isnan(alphas)):
            raise ValueError('Some alphas are still unmasked and NaN. Please file a bug report')
        return indices, alphas

    def _apply_boundary_conditions(self,
                                   indices,
                                   alphas,
                                   surface_index=None,
                                   bottom_index=None,
                                   surface_boundary_condition=None,
                                   bottom_boundary_condition=None):
        '''
        Applies the boundary conditions to the indices and alphas
        indices is expected to be fresh from the np.digitize step, meaning values
        from 0 to num_levels are expected.
        alphas is expected to still contain nans, but this function can still work by
        masking and setting values to 0 or 1 depending on the boundary condition
        '''
        surface_index = self.surface_index if surface_index is None else surface_index
        bottom_index = self.bottom_index if bottom_index is None else bottom_index
        surface_boundary_condition = self.default_surface_boundary_condition if surface_boundary_condition is None else surface_boundary_condition
        bottom_boundary_condition = self.default_bottom_boundary_condition if bottom_boundary_condition is None else bottom_boundary_condition

        if surface_index == 0:
            #ascending ordered depths (FVCOM-like) (0, 10, 20, ...)
            above_surf_mask = indices < surface_index
            below_bottom_mask = indices >= bottom_index
            alphas[above_surf_mask] = 1
            alphas[below_bottom_mask] = 0
        else:
            #descending ordered depths (ROMS-like) (..., 30, 20, 10, 0)
            above_surf_mask = indices >= surface_index
            below_bottom_mask = indices < bottom_index
            alphas[above_surf_mask] = 0
            alphas[below_bottom_mask] = 1
        oob_mask = np.logical_or(above_surf_mask, below_bottom_mask)
        indices.mask = np.logical_or(indices.mask, oob_mask)
        alphas.mask = np.logical_or(alphas.mask, oob_mask)

        if surface_boundary_condition == 'extrapolate':
            indices.mask[above_surf_mask] = False
            alphas.mask[above_surf_mask] = False
        if bottom_boundary_condition == 'extrapolate':
            indices.mask[below_bottom_mask] = False
            alphas.mask[below_bottom_mask] = False




        return indices, alphas, oob_mask




class ROMS_Depth(S_Depth):
    '''
    Sigma coordinate depth object for ROMS style output
    '''
    _instance_count = 0
    # hack so that older code can access _def_count
    @property
    def _def_count(self):
        return self._instance_count
    @_def_count.setter
    def _def_count(self, count):
        self.__class__._instance_count = count

    default_names = {'Cs_r': ['Cs_r'],
                     'Cs_w': ['Cs_w'],
                     's_rho': ['s_rho'],
                     's_w': ['s_w'],
                     'hc': ['hc'],
                     'bathymetry': ['h'],
                     'zeta': ['zeta']
                     }

    cf_names = {'Cs_r': ['S-coordinate stretching curves at RHO-points'],
                'Cs_w': ['S-coordinate stretching curves at W-points'],
                's_rho': ['S-coordinate at RHO-points'],
                's_w':['S-coordinate at W-points'],
                'hc': ['S-coordinate parameter, critical depth'],
                'bathymetry': ['bathymetry at RHO-points', 'Final bathymetry at RHO-points'],
                'zeta': ['free-surface']
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

    def get_transect(self, points, time, data_shape=None, _hash=None, **kwargs):
        '''
        :param points: array of points to interpolate to
        :type points: numpy array of shape (n, 3)

        :param time: time to interpolate to
        :type time: datetime.datetime

        :param data_shape: shape of the variable to be interpolated. This param is used to determine
        whether to index on the sigma layers or levels
        :type data_shape: tuple of int

        :return: numpy array of shape (n, num_w_levels) of n s-coordinate transects
        '''
        if data_shape is None:
            data_shape = (self.num_levels, )

        s_c = self.s_rho if data_shape[0] == self.num_layers else self.s_w
        C_s = self.Cs_r if data_shape[0] == self.num_layers else self.Cs_w
        h = self.bathymetry.at(points, time, unmask=False, _hash=_hash, **kwargs)
        zeta = self.zeta.at(points, time, unmask=False, _hash=_hash, **kwargs)
        hc = self.hc
        hCs = h * C_s[np.newaxis, :]
        if self.vtransform == 1:
            S = (hc* s_c) + hCs - (hc * C_s)[np.newaxis, :]
            s_coord = -(S + zeta * (1 + S / h))
        elif self.vtransform == 2:
            S = ((hc * s_c) + hCs) / (hc + h)
            s_coord = -(zeta + (zeta + h) * S)
        return s_coord


class FVCOM_Depth(S_Depth):
    _instance_count = 0

    # hack so that older code can access _def_count
    @property
    def _def_count(self):
        return self._instance_count
    @_def_count.setter
    def _def_count(self, count):
        self.__class__._instance_count = count

    default_names = {
        'siglay': ['siglay'], # mid layer depth coordinate on nodes
        'siglay_center': ['siglev_center'], # mid layer depth coordinate on centers
        'siglev': ['siglev'], # layer depth coordinate on nodes
        'siglev_center': ['siglev_center'], # layer depth coordinate on centers
        'bathymetry': ['h'], # bathymetry on nodes
        'h_center': ['h_center'], # bathymetry on centers
        'zeta': ['zeta'], # free surface
    }

    cf_names = {
        'siglay': ['ocean_sigma/general_coordinate'],
        'siglay_center': ['ocean_sigma/general_coordinate'],
        'siglev': ['ocean_sigma/general_coordinate'],
        'siglev_center': ['ocean_sigma/general_coordinate'],
        'bathymetry': ['sea_floor_depth_below_geoid'],
        'h_center': ['sea_floor_depth_below_geoid'],
        'zeta': ['sea_surface_height_above_geoid'],
    }

    @property
    def surface_index(self):
        return np.argmax(self.siglev[:,0])

    @property
    def bottom_index(self):
        return np.argmin(self.siglev[:,0])
    @property
    def num_levels(self):
        return len(self.siglev)

    @property
    def num_layers(self):
        return len(self.siglay)

    def get_transect(self, points, time, data_shape=None, _hash=None, **kwargs):
        '''
        :param points: array of points to interpolate to
        :type points: numpy array of shape (n, 3)

        :param time: time to interpolate to
        :type time: datetime.datetime

        :param data_shape:  Describes the shape of the data to be interpolated. 
        If the first dimension is the number of layers or if None, then siglay is used. 
        If the first dimension is the number of levels, then siglev is used.
        :type data_shape: tuple of int or None

        :return: numpy array of shape (n, num_w_levels) of n s-coordinate transects
        '''

        #because FVCOM sigma is defined for every node separately.
        sigvar = None
        if data_shape is None:
            sigvar = self.siglev[:].T
        elif data_shape[0] == self.num_layers:
            sigvar = self.siglay[:].T
        else:
            sigvar = self.siglev[:].T
        sigma = self.grid.interpolate_var_to_points(points[:, 0:2], sigvar, location='node')
        

        bathy = self.bathymetry.at(points, time, unmask=False, _hash=_hash, **kwargs)
        zeta = self.zeta.at(points, time, unmask=False, _hash=_hash, **kwargs)

        transects = -(zeta + (zeta + bathy) * sigma)
        return transects

        # elif self.vtransform == 2:
        #     S = ((hc * s_c) + hCs) / (hc + h)
        #     s_coord = -(zeta + (zeta + h) * S)
        # if no stretching or crit depth (hc, Cs_r, Cs_w) then S = s_c


class Depth():
    '''
    Factory class that generates depth objects.

    Also handles common loading and parsing operations
    '''
    ld_types = [L_Depth]
    sd_types = [ROMS_Depth, FVCOM_Depth]
    surf_types = [DepthBase]

    def __init__(self):
        raise NotImplementedError("Depth is not meant to be instantiated. "
                                  "Please use the 'from_netCDF' or 'surface_only' function")

    @staticmethod
    def surface_only(surface_index=-1,
                     **kwargs):
        '''
        If instantiated directly, this will always return a DepthBase It is assumed index -1
        is the surface index
        '''
        return DepthBase(surface_index=surface_index,
                         **kwargs)

    @classmethod
    def from_netCDF(cls,
                    filename=None,
                    dataset=None,
                    data_file=None,
                    grid_file=None,
                    **kwargs):
        '''
        :param filename: File containing a depth
        :type filename: string or list of string

        :param dataset: Takes precedence over filename, if provided.
        :type dataset: netCDF4.Dataset

        :param depth_type: Must be provided if autodetection is not possible.
            See Depth.ld_names, Depth.sd_names, and Depth.surf_names for the
            expected values for this argument
        :type depth_type: string

        :returns: Instance of L_Depth or S_Depth
        '''
        ds, dg = parse_filename_dataset_args(filename=filename,
                                             dataset=dataset,
                                             data_file=data_file,
                                             grid_file=grid_file)

        typs = cls.sd_types + cls.ld_types
        available_to_create = [typ._can_create_from_netCDF(grid_file=dg, data_file=ds) for typ in typs]
        if not any(available_to_create):
            warnings.warn('''Unable to automatically determine depth system so
                            reverting to surface-only mode. Manually check the
                            (depth_object).surface_index attribute and set it
                            to the appropriate array index for your model data''', RuntimeWarning)
            return cls.surf_types[0].from_netCDF(data_file=ds, grid_file=dg, **kwargs)
        else:
            typ = typs[np.argmax(available_to_create)]
            if sum(available_to_create) > 1:
                warnings.warn('''Multiple depth systems detected. Using the first one found: {0}'''.format(typ.__repr__), RuntimeWarning)
            return typ.from_netCDF(filename=filename,
                                   dataset=dataset,
                                   data_file=data_file,
                                   grid_file=grid_file,
                                   **kwargs)
