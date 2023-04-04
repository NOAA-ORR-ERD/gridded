
import warnings

import numpy as np
from gridded.time import Time
from gridded.grids import Grid
from gridded.utilities import (get_dataset,
                               search_dataset_for_any_long_name,
                               search_dataset_for_variables_by_longname,
                               search_dataset_for_variables_by_varname,
                               merge_var_search_dicts
)



class DepthBase(object):
    _default_component_types = {'time': Time,
                                'grid': Grid,
                                'variable': None}

    def __init__(self,
                 surface_index=None,
                 bottom_index=None,
                 positive_down=True,
                 **kwargs):
        '''
        :param surface_index: array index of 'highest' level (closest to sea level)
        :param bottom_index: array index of 'lowest' level (closest to seafloor)
        :param positive_down: orientation of points coordinates
                              (for interpolation functions)
        '''
        self.surface_index = surface_index
        self.bottom_index = bottom_index

    @classmethod
    def from_netCDF(cls,
                    surface_index=-1,
                    **kwargs):
        return cls(surface_index,
                   **kwargs)

    def interpolation_alphas(self, points, time, data_shape, _hash=None):
        return (None, None)

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
    _def_count=0
    default_terms = [('depth_levels', 'depth')]
    default_names = {'depth': ['depth','depth_levels']}
    cf_names = {'depth': 'depth'}

    def __init__(self,
                 name=None,
                 terms=None,
                 surface_index=None,
                 bottom_index=None,
                 **kwargs):
        super(L_Depth, self).__init__(**kwargs)
        self.name=name
        self.terms={}
        if terms is None:
            raise ValueError('Must provide terms for level depth coordinate')
        else:
            self.terms=terms
            for k, v in terms.items():
                setattr(self, k, v)
        self.surface_index = surface_index
        self.bottom_index = bottom_index

    @classmethod
    def from_netCDF(cls,
                    filename=None,
                    dataset=None,
                    name=None,
                    topology=None,
                    terms=None,
                    surface_index=None,
                    bottom_index=None,
                    **kwargs
                    ):
        df = dataset if filename is None else get_dataset(filename, dataset)

        if df is None:
            raise ValueError('No filename or dataset provided')
        if name is None:
            name = cls.__name__ + str(cls._def_count)
        if terms is None:
            terms={}
            for tn, tln in cls.default_terms:
                vname=tn
                #if tn not in dataset.variables.keys(): # 2023
                if tn not in df.variables.keys():
                    vname = cls._gen_varname(filename, dataset, [tn], [tln])
                #terms[tn] = dataset[vname][:] # 2023
                terms[tn] = df[vname][:]
        if surface_index is None:
            surface_index = np.argmin(terms['depth_levels'])
        if bottom_index is None:
            bottom_index = np.argmax(terms['depth_levels'])
        # 2023-02-21 set the depth of the top layer to zero
        terms['depth_levels'][surface_index] = 0.0
        # 2023-02-21 set the depth of the top layer to zero

        return cls(name=name,
                   terms=terms,
                   surface_index=surface_index,
                   bottom_index=bottom_index,
                   **kwargs)

    def interpolation_alphas(self, points, *args, **kwargs):
        '''
        Returns a pair of values. The 1st value is an array of the depth indices of all the particles.
        The 2nd value is an array of the interpolation alphas for the particles between their depth
        index and depth_index+1. If both values are None, then all particles are on the surface layer.
        '''
        points = np.asarray(points, dtype=np.float64)
        points = points.reshape(-1, 3)
        withingrid = np.logical_and(points[:, 2] >= 0, points[:, 2] <= np.max(self.depth_levels))
        abovewater = points[:, 2] <0
        belowgrid = points[:, 2] > np.max(self.depth_levels)
        onsurface = points[:, 2]  == 0

        if len(np.where(onsurface)[0]) == len(points):
            return None, None

        indices = -np.ones((len(points)), dtype=np.int64)
        alphas = -np.ones((len(points)), dtype=np.float64)
        pts = points[withingrid]
        und_ind = -np.ones((len(np.where(withingrid)[0])))
        und_alph = und_ind.copy()

        und_ind = np.digitize(pts[:,2], self.depth_levels, right=True)

        for i,n in enumerate(und_ind):
            if n == len(self.depth_levels):
                und_ind[i] = n-1
                und_alph[i] = 1.0
            elif n == 0:
                und_ind[i] = 1
                und_alph[i] = 0.0
            else:
                und_alph[i] = (pts[i, 2] - self.depth_levels[und_ind[i]-1]) / (
                    self.depth_levels[und_ind[i]] - self.depth_levels[und_ind[i]-1])
        indices[withingrid] = und_ind
        alphas[withingrid] = und_alph

        indices[belowgrid] = -1
        alphas[belowgrid] = -2

        indices[abovewater] = -1
        alphas[abovewater] = -3

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
    Represents the ocean s-coordinate, with particular focus on ROMS implementation
    '''
    _def_count = 0
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

    def __init__(self,
                 name=None,
                 time=None,
                 grid=None,
                 bathymetry=None,
                 zeta=None,
                 terms=None,
                 vtransform=2,
                 positive_down=True,
                 zero_ref = 'absolute',
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

        :param positive_down: Flag for interpreting depth values as positive or negative
        This applies to any coordinates passed into the various functions, NOT the values
        of the S-Coordinates that this object represents
        :type positive_down: boolean

        :param zero_ref: Determines whether the 0 datum moves with the surface or is fixed
        :type zero_ref: string ('absolute' or 'surface')
        '''

        super(S_Depth, self).__init__(**kwargs)
        self.name=name
        self.time=time
        self.grid=grid
        self.bathymetry = bathymetry
        self.zeta = zeta
        self.terms={}
        self.vtransform = vtransform
        self.positive_down = positive_down
        self.zero_ref = zero_ref
        if terms is None:
            raise ValueError('Must provide terms for sigma coordinate')
        else:
            self.terms=terms
            for k, v in terms.items():
                setattr(self, k, v)
        if self.surface_index is None:
            self.surface_index = self.num_w_levels - 1
        if self.bottom_index is None:
            self.bottom_index = 0


    @classmethod
    def from_netCDF(cls,
                    filename=None,
                    varnames=None,
                    grid_topology=None,
                    name=None,
                    #units=None,
                    time=None,
                    #time_origin=None,
                    grid=None,
                    dataset=None,
                    data_file=None,
                    grid_file=None,
                    #load_all=False,
                    bathymetry=None,
                    zeta=None,
                    terms=None,
                    #fill_value=0,
                    vtransform=2,
                    positive_down=True,
                    **kwargs
                    ):
        '''
        :param filename: A string or ordered list of string of netCDF filename(s)
        :type filename: string or list
        :param varnames: Direct mapping of component name to netCDF variable name. Use
            this if auto detection fails. Partial definition is allowable. Unspecified
            terms will use the value in `.default_names`
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
        :type name: string
        :param time: Time dimension (for zeta)
        :type time: gridded.time.Time or subclass
        :param grid: X-Y dmension (for bathymetry & zeta)
        :type grid: subclass of gridded.grids.GridBase
        '''
        Grid = cls._default_component_types['grid']
        Time = cls._default_component_types['time']
        Variable = cls._default_component_types['variable']
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
        if name is None:
            name = cls.__name__ + str(cls._def_count)
            cls._def_count += 1
        varnames = cls.default_names.copy()
        
        #Do a comprehensive search for netCDF4 Variables all at once
        vn_search = search_dataset_for_variables_by_varname(ds, varnames)
        ds_search = search_dataset_for_variables_by_longname(ds, cls.cf_names)
        varnames = merge_var_search_dicts(ds_search, vn_search)
        if ds != dg:
            dg_search = search_dataset_for_variables_by_longname(dg, cls.cf_names)
            varnames = merge_var_search_dicts(varnames, dg_search)
        
        if bathymetry is None:
            bathy_var = varnames.get('bathymetry', None)
            if bathy_var is None:
                err = 'Unable to locate bathymetry in data file'
                if dg:
                    raise ValueError(err + ' or grid file')
                raise ValueError(err)
            bathymetry = Variable.from_netCDF(dataset=bathy_var._grp,
                                              varname=bathy_var.name,
                                              grid=grid,
                                              name='bathymetry',
                                              )

        if zeta is None:
            zeta_var = varnames.get('zeta', None)
            if zeta_var is None:
                warn = 'Unable to locate zeta in data file'
                if dg:
                    warnings.warn(warn + ' or grid file.')
                warn += ' Generating constant (0) zeta.'
                warnings.warn(err)
                zeta = Variable.constant(0)
            else:
                zeta = Variable.from_netCDF(dataset=zeta_var._grp,
                                            varname=zeta_var.name,
                                            grid=grid,
                                            name='zeta')
                
        if time is None:
            time = zeta.time
        if terms is None:
            terms = {}
            for term, tvar in varnames.items():
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
                   positive_down=positive_down,
                   **kwargs)

    @property
    def num_w_levels(self):
        return len(self.s_w)

    @property
    def num_r_levels(self):
        return len(self.s_rho)

    def __len__(self):
        return self.num_w_levels

    def _L_Depth_given_bathymetry_t1(self, bathy, zeta, lvl, rho_or_w):
        """
        computes the depth (positive up) of a level given the bathymetry
        and zeta. Produces an output array same shape as bathy and zeta
        Output is always 'positive down'
        This implements transform 1 (https://www.myroms.org/wiki/Vertical_S-coordinate)
        """
        if rho_or_w == 'rho':
            s = self.s_rho[lvl]
            Cs = self.Cs_r[lvl]
        elif rho_or_w == 'w':
            s = self.s_w[lvl]
            Cs = self.Cs_w[lvl]
        else:
            raise ValueError('invalid rho_or_w argument (must be "rho" or "w")')

        hc = self.hc
        S = hc * s + (bathy - hc) * Cs
        return -(S + zeta * (1 + S / bathy))

    def _L_Depth_given_bathymetry_t2(self, bathy, zeta, lvl, rho_or_w):
        """
        computes the depth (positive up) of a level given the bathymetry
        and zeta. Produces an output array same shape as bathy and zeta
        Output is always 'positive down'
        This implements transform 2 (https://www.myroms.org/wiki/Vertical_S-coordinate)
        """
        if rho_or_w == 'rho':
            s = self.s_rho[lvl]
            Cs = self.Cs_r[lvl]
        elif rho_or_w == 'w':
            s = self.s_w[lvl]
            Cs = self.Cs_w[lvl]
        else:
            raise ValueError('invalid rho_or_w argument (must be "rho" or "w")')

        hc = self.hc
        S = (hc * s + bathy * Cs) / (hc + bathy)
        return -(zeta + (zeta + bathy) * S)

    def interpolation_alphas(self, points, time, data_shape, _hash=None, extrapolate=False):
        '''
        Returns a pair of values. The 1st value is an array of the depth indices
        of all the points. The 2nd value is an array of the interpolation
        alphas for the points between their depth index and depth_index+1. If
        both values are None, then all points are on the surface layer.

        zero_ref can be 'surface' or 'absolute'.
            'surface' means a point's depth is referenced from zeta
            'absolute' means a point's depth is referenced from average sea level zero
        '''
        zetas = self.zeta.at(points, time, _hash=_hash, extrapolate=extrapolate).reshape(-1)
        #abs_depths = absolute depth from avg sea level, positive = down
        #below_surface = bool array flagging a particle as below sea surface and above sea floor
        #below_ground = bool array flagging a particle as below sea floor (abs_depth > bathymetry)
        abs_depths = below_surface = below_ground = None
        if self.positive_down: #points depths are positive down
            abs_depths = points[:,2]
        else: #points depths are positive up
            abs_depths = -points[:,2]
        if self.zero_ref == 'absolute':
            below_surface = -abs_depths < zetas.reshape(-1)
        else:
            abs_depths = abs_depths - zetas
            below_surface = points[:,2] > 0

        if not np.any(below_surface):
            # nothing is underwater, so return special (None, None)
            return None, None
        bathy = self.bathymetry.at(points, time, _hash=_hash,
                                   extrapolate=extrapolate).reshape(-1)
        below_ground = abs_depths > bathy

        # below_surface points should not also be below_ground
        # np.logical_and(np.logical_not(below_ground), below_surface, below_surface)

        # setup (index, alphas) return arrays. -1 indicates at or above surface
        indices = -np.ones((len(points)), dtype=np.int64)
        alphas = -np.ones((len(points)), dtype=np.float64)

        bs_zeta = zetas[below_surface] # len(bs_zeta) = # of True in below_surface\
        bs_bathy = bathy[below_surface] # only a few particles may be below surface
        bs_depths = abs_depths[below_surface]

        # working arrays for below_surface points
        und_ind = -np.ones((len(bs_zeta)))
        und_alph = und_ind.copy()

        rho_or_w = None #parameter necessary for ldgb
        if data_shape[0] == self.num_w_levels:
            #data assumed to be on w points
            rho_or_w = 'w'
        elif data_shape[0] == self.num_r_levels:
            #data assumed to be on rho points
            rho_or_w = 'rho'
        else:
            raise ValueError('Cannot get depth interpolation alphas for data shape specified; does not fit r or w depth axis')

        if self.vtransform == 2:
            ldgb = self._L_Depth_given_bathymetry_t2
        elif self.vtransform == 1:
            ldgb = self._L_Depth_given_bathymetry_t1
        else:
            raise ValueError('invalid vtransform attribute on depth object')
        #blev_depths = level depth below the position, ulev_depths = level depth above the position
        blev_depths = ulev_depths = None
        b_index = self.bottom_index
        t_index = self.surface_index if rho_or_w == 'rho' else self.surface_index + 1
        #this loop finds the indices one level at a time.
        for level in range(b_index, t_index, 1):
            # for the current level, get the level depths given bathymetry and zeta
            # start at level 0 (deepest)
            ulev_depths = ldgb(bs_bathy, bs_zeta, level, rho_or_w)

            # print(ulev_depths)
            below_upper_layer = np.logical_and(ulev_depths < bs_depths, und_ind >= -1)
            if level == b_index: #special case for lowest possible layer
                und_ind[below_upper_layer] = -2
                und_alph[below_upper_layer] = -2
                continue
            blev_depths = ldgb(bs_bathy, bs_zeta, level-1, rho_or_w)
            within_layer = np.where(np.logical_and(blev_depths >= bs_depths, below_upper_layer))[0]
            if len(within_layer) == 0:
                continue
            # print(within_layer)
            und_ind[within_layer] = level
            a = ((bs_depths.take(within_layer) - blev_depths.take(within_layer)) /
                    (ulev_depths.take(within_layer) - blev_depths.take(within_layer)))
            und_alph[within_layer] = a
            blev_depths = ulev_depths

        indices[below_surface] = und_ind
        alphas[below_surface] = und_alph
        indices[indices == -2] = -1
        return indices, alphas

    def get_section(self, time, coord='w', x_coord=None, y_coord=None, vtransform=2):
        '''
        Returns a section  of the z-level space in time. All s-levels are
        returned in the data. By providing a x_coord or y_coord you can
        get cross sections in the direction specified, or both for the level
        depths at a single point.
        '''
        if coord not in ['w', 'rho']:
            raise ValueError('Can only specify "w" or "rho" for coord kwarg')
        ldgb = None
        z_shp = None
        if vtransform == 2:
            ldgb = self._L_Depth_given_bathymetry_t2
        else:
            ldgb = self._L_Depth_given_bathymetry_t1

        if coord == 'w':
            z_shp = (self.num_w_levels, self.zeta.data.shape[-2], self.zeta.data.shape[-1])
        if coord == 'rho':
            z_shp = (self.num_r_levels, self.zeta.data.shape[-2], self.zeta.data.shape[-1])
        time_idx = self.time.index_of(time)
        time_alpha = self.time.interp_alpha(time)
        z0  = self.zeta.data[time_idx]
        if time_idx == len(self.time.data) - 1 or time_idx == 0:
            zeta = self.zeta.data[time_idx]
        else:
            z1 = self.zeta.data[time_idx + 1]
            zeta = (z1 - z0)*(1-time_alpha) + z0
        bathy = self.bathymetry.data[:]
        z_data=np.empty(z_shp)
        for lvl in range(0,len(z_data)):
            z_data[lvl] = ldgb(bathy, zeta, lvl, coord)
        return z_data


class Depth(object):
    '''
    Factory class that generates depth objects. Also handles common loading and
    parsing operations
    '''
    ld_names = ['level', 'levels', 'L_Depth', 'depth_levels' 'depth_level']
    sd_names = ['sigma']
    surf_names = ['surface', 'surface only', 'surf', 'none']
    def __init__(self):
        raise NotImplementedError("Depth is not meant to be instantiated. "
                                  "Please use the 'from_netCDF' or 'surface_only' function")
    @staticmethod
    def surface_only(*args, **kwargs):
        '''
        If instantiated directly, this will always return a DepthBase It is assumed index -1
        is the surface index
        '''
        return DepthBase(*args,**kwargs)

    @staticmethod
    def from_netCDF(filename=None,
                    dataset=None,
                    data_file=None,
                    grid_file=None,
                    depth_type=None,
                    varname=None,
                    topology=None,
                    _default_types=(('level', L_Depth), ('sigma', S_Depth), ('surface', DepthBase)),
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

        df = dataset if filename is None else get_dataset(filename, dataset)
        if df is None:
            raise ValueError('No filename or dataset provided')

        cls = depth_type
        if (depth_type is None or isinstance(depth_type, str) or
                                 not issubclass(depth_type, DepthBase)):
            cls = Depth._get_depth_type(df, depth_type, topology, _default_types)

        return cls.from_netCDF(filename=filename,
                               dataset=dataset,
                               grid_file=grid_file,
                               data_file=data_file,
                               topology=topology,
                               **kwargs)

    # @staticmethod
    # def depth_type_of_var(dataset, varname):
    #     '''
    #     Given a varname or netCDF variable and dataset, try to determine the
    #     depth type from dimensions, attributes, or other metadata information,
    #     and return a grid_type and topology
    #     '''
    #     var = dataset[varname]

    @staticmethod
    def _get_depth_type(dataset, depth_type=None, topology=None, _default_types=None):
        if _default_types is None:
            _default_types = dict()
        else:
            _default_types = dict(_default_types)

        S_Depth = _default_types.get('sigma', None)
        L_Depth = _default_types.get('level', None)
        Surface_Depth = _default_types.get('surface', None)
        if depth_type is not None:
            if depth_type.lower() in Depth.ld_names:
                return L_Depth
            elif depth_type.lower() in Depth.sd_names:
                return S_Depth
            elif depth_type.lower() in Depth.surf_names:
                return Surface_Depth
            else:
                raise ValueError('Specified depth_type not recognized/supported')
        if topology is not None:
            if ('faces' in topology.keys()
                    or topology.get('depth_type', 'notype').lower() in ld_names):
                return L_Depth
            elif topology.get('depth_type', 'notype').lower() in Depth.sd_names:
                return S_Depth
            else:
                return Surface_Depth

        
        else:
            t1 = search_dataset_for_any_long_name(dataset, S_Depth.cf_names)
            if t1:
                return S_Depth
            else:
                try:
                    L_Depth.from_netCDF(dataset=dataset)
                    return L_Depth
                except:
                    warnings.warn('''Unable to automatically determine depth system so reverting to surface-only mode. Please verify the index of the surface is correct.''', RuntimeWarning)
                    return Surface_Depth
