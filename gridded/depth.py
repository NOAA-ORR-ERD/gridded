from __future__ import absolute_import, division, print_function, unicode_literals

import warnings
from six import string_types

import gridded
import numpy as np
from datetime import datetime
from gridded.time import Time
from gridded.grids import Grid
from gridded.utilities import get_dataset

class DepthBase(object):
    _default_component_types = {'time': Time,
                                'grid': Grid,
                                'variable': None}
    def __init__(self,
                 surface_index=None,
                 bottom_index=None,
                 **kwargs):
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
        raise ValueError("Default names not found.")


class L_Depth(DepthBase):
    _def_count=0
    default_terms = [('depth_levels','depth')]

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
                if tn not in dataset.variables.keys():
                    vname = cls._gen_varname(filename, dataset, [tn], [tln])
                terms[tn] = dataset[vname][:]
        if surface_index is None:
            surface_index = np.argmin(terms['depth_levels'])
        if bottom_index is None:
            bottom_index = np.argmax(terms['depth_levels'])
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
        underwater = points[:, 2] > 0
        if len(np.where(underwater)[0]) == 0:
            return None, None
        indices = -np.ones((len(points)), dtype=np.int64)
        alphas = -np.ones((len(points)), dtype=np.float64)
        pts = points[underwater]
        und_ind = -np.ones((len(np.where(underwater)[0])))
        und_alph = und_ind.copy()
        und_ind = np.digitize(pts[:,2], self.depth_levels) - 1
        for i,n in enumerate(und_ind):
            if n == len(self.depth_levels) -1:
                und_ind[i] = -1
            if und_ind[i] != -1:
                und_alph[i] = (pts[i,2] - self.depth_levels[und_ind[i]]) / (self.depth_levels[und_ind[i]+1] - self.depth_levels[und_ind[i]])
        indices[underwater] = und_ind
        alphas[underwater] = und_alph
        return indices, alphas


    @classmethod
    def _find_required_depth_attrs(cls, filename, dataset=None, topology=None):

        # THESE ARE ACTUALLY ALL OPTIONAL. This should be migrated when optional attributes are dealt with
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
    _def_count=0
    default_terms = [('Cs_r', 'S-coordinate stretching curves at RHO-points'),
                     ('Cs_w', 'S-coordinate stretching curves at W-points'),
                     ('s_rho', 'S-coordinate at RHO-points'),
                     ('s_w', 'S-coordinate at W-points'),
                     ('h', 'bathymetry at RHO-points'),
                     ('hc', 'S-coordinate parameter, critical depth'),
                     ('zeta', 'free-surface')]

    def __init__(self,
                 name=None,
                 time=None,
                 grid=None,
                 bathymetry=None,
                 zeta=None,
                 terms=None,
                 **kwargs):
        super(S_Depth, self).__init__(**kwargs)
        if self.surface_index is None:
            self.surface_index = -1
        if self.bottom_index is None:
            self.bottom_index = 0
        self.name=name
        self.time=time
        self.grid=grid
        self.bathymetry = bathymetry
        self.zeta = zeta
        self.terms={}
        if terms is None:
            raise ValueError('Must provide terms for sigma coordinate')
        else:
            self.terms=terms
            for k, v in terms.items():
                setattr(self, k, v)


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
                    dataset=None,
                    data_file=None,
                    grid_file=None,
                    load_all=False,
                    bathymetry=None,
                    zeta=None,
                    terms=None,
                    fill_value=0,
                    **kwargs
                    ):
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
        if bathymetry is None:
            bathy_name = cls._gen_varname(filename=filename,
                                           dataset=ds,
                                           names_list=['h'],
                                           std_names_list=['bathymetry at RHO-points'])
            bathymetry = Variable.from_netCDF(dataset=ds,
                                  grid=grid,
                                  varname=bathy_name,
                                  name='Bathymetry'
                                  )
        if zeta is None:
            zeta_name = cls._gen_varname(filename=filename,
                                           dataset=ds,
                                           names_list=['zeta'],
                                           std_names_list=['free-surface'])
            zeta = Variable.from_netCDF(dataset=ds,
                                  grid=grid,
                                  varname=zeta_name,
                                  name='zeta'
                                  )
        if time is None:
            time = zeta.time
        if terms is None:
            terms={}
            for tn, tln in cls.default_terms:
                vname=tn
                if tn not in ds.variables.keys():
                    vname = cls._gen_varname(filename, ds, [tn], [tln])
                if tn not in ['h','zeta']: #don't want to reinclude bathymetry
                    terms[vname] = ds[vname][:]
        return cls(name=name,
                   time=time,
                   grid=grid,
                   bathymetry=bathymetry,
                   zeta=zeta,
                   terms=terms,
                   **kwargs)

    @property
    def num_w_levels(self):
        return len(self.s_w)

    @property
    def num_r_levels(self):
        return len(self.s_rho)

    def __len__(self):
        return self.num_w_levels

    def _w_L_Depth_given_bathymetry(self, depths, zeta, lvl):
        s_w = self.s_w[lvl]
        Cs_w = self.Cs_w[lvl]
        hc = self.hc
        S = hc * s_w + (depths - hc) * Cs_w
        return -(S + zeta * (1 + S / depths))

    def _r_L_Depth_given_bathymetry(self, depths, zeta, lvl):
        s_rho = self.s_rho[lvl]
        Cs_r = self.Cs_r[lvl]
        hc = self.hc
        S = hc * s_rho + (depths - hc) * Cs_r
        return -(S + zeta * (1 + S / depths))

    def interpolation_alphas(self, points, time, data_shape, _hash=None):
        '''
        Returns a pair of values. The 1st value is an array of the depth indices of all the particles.
        The 2nd value is an array of the interpolation alphas for the particles between their depth
        index and depth_index+1. If both values are None, then all particles are on the surface layer.
        '''
        underwater = points[:, 2] > -self.zeta.at(points, time)
        if len(np.where(underwater)[0]) == 0:
            return None, None
        indices = -np.ones((len(points)), dtype=np.int64)
        alphas = -np.ones((len(points)), dtype=np.float64)
        depths = self.bathymetry.at(points, time, _hash=_hash)[underwater]
        zeta = self.zeta.at(points, time, _hash=_hash)[underwater]
        pts = points[underwater]
        und_ind = -np.ones((len(np.where(underwater)[0])))
        und_alph = und_ind.copy()

        if data_shape[0] == self.num_w_levels:
            num_levels = self.num_w_levels
            ldgb = self._w_L_Depth_given_bathymetry
        elif data_shape[0] == self.num_r_levels:
            num_levels = self.num_r_levels
            ldgb = self._r_L_Depth_given_bathymetry
        else:
            raise ValueError('Cannot get depth interpolation alphas for data shape specified; does not fit r or w depth axis')
        blev_depths = ulev_depths = None
        for ulev in range(0, num_levels):
            ulev_depths = ldgb(depths, zeta, ulev)
#             print ulev_depths[0]
            within_layer = np.where(np.logical_and(ulev_depths < pts[:, 2], und_ind == -1))[0]
#             print within_layer
            und_ind[within_layer] = ulev
            if ulev == 0:
                und_alph[within_layer] = -2
            else:
                a = ((pts[:, 2].take(within_layer) - blev_depths.take(within_layer)) /
                     (ulev_depths.take(within_layer) - blev_depths.take(within_layer)))
                und_alph[within_layer] = a
            blev_depths = ulev_depths

        indices[underwater] = und_ind
        alphas[underwater] = und_alph
        return indices, alphas

    def get_section(self, time, coord='w', x_coord=None, y_coord=None, ):
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
        if coord == 'w':
            ldgb = self._w_L_Depth_given_bathymetry
            z_shp = (self.num_w_levels, self.zeta.data.shape[-2], self.zeta.data.shape[-1])
        if coord == 'rho':
            ldgb = self._r_L_Depth_given_bathymetry
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
            z_data[lvl] = ldgb(bathy,zeta,lvl)
        return z_data


class Depth(object):
    '''
    Factory class that generates depth objects. Also handles common loading and
    parsing operations
    '''
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
                    depth_type=None,
                    varname=None,
                    topology=None,
                    _default_types=(('level', L_Depth), ('sigma', S_Depth), ('surface', DepthBase)),
                    *args,
                    **kwargs):
        '''
        :param filename: File containing a depth
        :param dataset: Takes precedence over filename, if provided.
        :param depth_type: Must be provided if autodetection is not possible.
        :returns: Instance of L_Depth or S_Depth
        '''
        df = dataset if filename is None else get_dataset(filename, dataset)
        if df is None:
            raise ValueError('No filename or dataset provided')

        cls = depth_type
        if (depth_type is None or isinstance(depth_type, string_types) or
                                 not issubclass(depth_type, DepthBase)):
            cls = Depth._get_depth_type(df, depth_type, topology, _default_types)

        return cls.from_netCDF(filename=filename,
                               dataset=dataset,
                               topology=topology,
                               **kwargs)

    @staticmethod
    def depth_type_of_var(dataset, varname):
        '''
        Given a varname or netCDF variable and dataset, try to determine the depth type from dimensions,
        attributes, or other metadata information, and return a grid_type and topology
        '''
        var = dataset[varname]


    @staticmethod
    def _get_depth_type(dataset, depth_type=None, topology=None, _default_types=None):
        if _default_types is None:
            _default_types = dict()
        else:
            _default_types = dict(_default_types)

        S_Depth = _default_types.get('sigma', None)
        L_Depth = _default_types.get('level', None)
        Surface_Depth = _default_types.get('surface', None)
        ld_names = ['level', 'levels', 'L_Depth', 'depth_levels' 'depth_level']
        sd_names = ['sigma']
        surf_names = ['surface', 'surface only', 'surf', 'none']
        if depth_type is not None:
            if depth_type.lower() in ld_names:
                return L_Depth
            elif depth_type.lower() in sd_names:
                return S_Depth
            elif depth_type.lower() in surf_names:
                return Surface_Depth
            else:
                raise ValueError('Specified depth_type not recognized/supported')
        if topology is not None:
            if ('faces' in topology.keys() or
                topology.get('depth_type', 'notype').lower() in ld_names):
                return L_Depth
            elif topology.get('depth_type', 'notype').lower() in sd_names:
                return S_Depth
            else:
                return Surface_Depth
        else:
            try:
                L_Depth.from_netCDF(dataset=dataset)
                return L_Depth
            except ValueError:
                try:
                    S_Depth.from_netCDF(dataset=dataset)
                    return S_Depth
                except ValueError:
                    warnings.warn('''Unable to automatically determine depth system so reverting to surface-only mode. Please verify the index of the surface is correct.''', RuntimeWarning)
                    return Surface_Depth