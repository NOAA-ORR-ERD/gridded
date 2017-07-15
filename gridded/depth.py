from __future__ import absolute_import, division, print_function, unicode_literals

# import gridded
import numpy as np
from datetime import datetime
from gridded.time import Time
from gridded.grids import Grid
from gridded.utilities import get_dataset
from gridded.variable import Variable

class Depth(object):

    def __init__(self,
                 surface_index=-1):
        self.surface_index = surface_index
        self.bottom_index = surface_index

    @classmethod
    def from_netCDF(cls,
                    surface_index=-1):
        return cls(surface_index)

    def interpolation_alphas(self, points, data_shape, _hash=None):
        return (None, None)


class S_Depth(Depth):
    '''
    Represents the ocean s-coordinate, with particular focus on ROMS implementation
    '''
    default_terms = [('Cs_r', 'S-coordinate stretching curves at RHO-points'),
                     ('Cs_w', 'S-coordinate stretching curves at W-points'),
                     ('s_rho', 'S-coordinate at RHO-points'),
                     ('s_w', 'S-coordinate at W-points'),
                     ('h', 'bathymetry at RHO-points'),
                     ('hc', 'S-coordinate parameter, critical depth'),
                     ('zeta', 'free-surface')]
    _default_component_types = {'time': Time,
                                'grid': Grid,
                                'variable': Variable}
    def __init__(self,
                 filename=None,
                 dataset=None,
                 time=None,
                 grid=None,
                 bathymetry=None,
                 **kwargs):
        ds = dataset
        if ds is None:
            if filename is None:
                raise ValueError("Need filename or dataset containing sigma equation terms")
            ds = get_dataset(filename)
        self.bathymetry = bathymetry
        if bathymetry is None:
            bathy_name = self._gen_varname(filename=filename,
                                           dataset=ds,
                                           names_list=['h'],
                                           std_names_list=['bathymetry at RHO-points'])
            bathymetry = Variable.from_netCDF(dataset=ds,
                                  grid=grid,
                                  varname=bathy_name,
                                  name='Bathymetry'
                                  )
        self.bathymetry = bathymetry
        self.terms={}


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
        if varnames is None:
            varnames = [vn[0] for vn in S_Depth.default_terms]
            for n in varnames:
                if n not in ds.variables.keys():
                    raise NameError('Cannot find term {0} in the dataset'.format(n))
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
        if time is None:
            time = Time.from_netCDF(filename=data_file,
                                    dataset=ds,
                                    datavar=bathymetry)
        return cls(filename=filename,
                   dataset=ds,
                   time=time,
                   grid=grid,
                   bathymetry=bathymetry)

    @property
    def surface_index(self):
        return

    @property
    def bottom_index(self):
        return 0

    @property
    def num_w_levels(self):
        return len(self.terms['s_w'])

    @property
    def num_r_levels(self):
        return len(self.terms['s_rho'])

    def _w_level_depth_given_bathymetry(self, depths, lvl):
        s_w = self.terms['s_w'][lvl]
        Cs_w = self.terms['Cs_w'][lvl]
        hc = self.terms['hc']
        return -(hc * (s_w - Cs_w) + Cs_w * depths)

    def _r_level_depth_given_bathymetry(self, depths, lvl):
        s_rho = self.terms['s_rho'][lvl]
        Cs_r = self.terms['Cs_r'][lvl]
        hc = self.terms['hc']
        return -(hc * (s_rho - Cs_r) + Cs_r * depths)

    def interpolation_alphas(self, points, data_shape, _hash=None):
        '''
        Returns a pair of values. The 1st value is an array of the depth indices of all the particles.
        The 2nd value is an array of the interpolation alphas for the particles between their depth
        index and depth_index+1. If both values are None, then all particles are on the surface layer.
        '''
        underwater = points[:, 2] > 0.0
        if len(np.where(underwater)[0]) == 0:
            return None, None
        indices = -np.ones((len(points)), dtype=np.int64)
        alphas = -np.ones((len(points)), dtype=np.float64)
        depths = self.bathymetry.at(points, datetime.now(), _hash=_hash)[underwater]
        pts = points[underwater]
        und_ind = -np.ones((len(np.where(underwater)[0])))
        und_alph = und_ind.copy()

        if data_shape[0] == self.num_w_levels:
            num_levels = self.num_w_levels
            ldgb = self._w_level_depth_given_bathymetry
        elif data_shape[0] == self.num_r_levels:
            num_levels = self.num_r_levels
            ldgb = self._r_level_depth_given_bathymetry
        else:
            raise ValueError('Cannot get depth interpolation alphas for data shape specified; does not fit r or w depth axis')
        blev_depths = ulev_depths = None
        for ulev in range(0, num_levels):
            ulev_depths = ldgb(depths, ulev)
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
