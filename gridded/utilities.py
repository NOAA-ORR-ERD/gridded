#!/usr/bin/env python

"""
assorted utility functions needed by gridded
"""

try:
    from collections.abc import Iterable, Collection, Mapping
except ImportError:  # py2
    from collections import Iterable

import numpy as np
import netCDF4 as nc4
import os


must_have = ['dtype', 'shape', 'ndim', '__len__', '__getitem__', '__getattribute__']

def convert_numpy_datetime64_to_datetime(dt):
    pass

def convert_mask_to_numpy_mask(mask_var):
    '''
    Converts a netCDF4.Variable representing a mask into a numpy array mask
    'Water' Values are converted to False (including 'lake', 'river' etc)
    'land' values are converted to True
    '''
    ret_mask = np.ones(mask_var.shape, dtype=bool)
    mask_data = mask_var[:]
    type1 = (isinstance(mask_var, nc4.Variable)
             and hasattr(mask_var, 'flag_values')
             and hasattr(mask_var, 'flag_meanings'))
    type2 = (isinstance(mask_var, nc4.Variable)
             and hasattr(mask_var, 'option_0'))
    if type1:
        fm = mask_var.flag_meanings
        fv = mask_var.flag_values
        try:
            fm = fm.split()
        except AttributeError:
            pass  # must not be a string -- we assume it's a sequence already
        try:
            fv = fv.split()
        except AttributeError:
            pass
        meaning_mask = [False if ('water' in s or 'lake' in s) else True for s in fm]
        tfmap = dict(zip(fv, meaning_mask))
        for k, v in tfmap.items():
            ret_mask[mask_data == k] = v
    elif type2:  # special case where option_0 == land,
                 # option_1 == water, etc
                 # TODO: generalize this properly
        meaning_mask = [True, False]
        tfmap = dict(zip([0, 1], meaning_mask))
        for k, v in tfmap.items():
            ret_mask[mask_data == k] = v
    else:
        ret_mask[:] = mask_data

    return ret_mask


def gen_celltree_mask_from_center_mask(center_mask, sl):
    """
    Generates celltree face mask from the center mask
    """

    input_mask = convert_mask_to_numpy_mask(center_mask)
    return input_mask[sl]

#1/16/2024 Jay Hennen: I'm 'privating' this function pending a review and likely rewrite. 
#
def regrid_variable(grid, o_var, location='node'):
    from gridded.variable import Variable
    from gridded.grids import Grid_S, Grid_U
    from gridded.depth import S_Depth, L_Depth, DepthBase
    """
    Takes a Variable or VectorVariable and interpolates the data onto grid.
    You may pass a location ('nodes', 'faces', 'edge1', 'edge2) and the
    variable will be interpolated there if possible
    If no location is passed, the variable will be interpolated to the
    nodes of this grid. If the Variable's grid and this grid are the same, this
    function will return the Variable unchanged.

    If this grid covers area that the source grid does not, all values
    in this area will be masked. If regridding from cell centers to the nodes,
    The values of any border point not within will be equal to the value at the
    center of the border cell.

    NOTE: This function will load the ENTIRE data space of the source Variable.
    Make sure that this is reasonable. If it is not, consider pre-slicing
    the time and depth dimensions of the source Variable to what you need.
    """

    dest_points = None
    if location == 'node':
        dest_points = grid.nodes
    if location == 'face' or location == 'center':
        if grid.face_coordinates is None and isinstance(grid, Grid_U):
            grid.build_face_coordinates()
            dest_points = grid.face_coordinates
        else:
            dest_points = grid.centers
    dest_points = dest_points.reshape(-1, 2)
    if 'edge' in location:
        raise NotImplementedError("Cannot regrid variable to edges at this time")
    dest_indices = o_var.grid.locate_faces(dest_points, 'node')
    if np.all(dest_indices == -1):
        raise ValueError("Grid {0} has no destination points overlapping\
        the grid of the source variable {1}".format(grid, o_var))
    n_depth = None
    if o_var.depth is not None:
        if issubclass(o_var.depth.__class__, S_Depth):
            n_depth = _regrid_s_depth(grid, o_var.depth)
        elif isinstance(o_var.depth, DepthBase) or isinstance(o_var.depth, L_Depth):
            n_depth = o_var.depth
        else:
            raise NotImplementedError("Can only regrid sigma depths for now")

    xy_shp = grid._get_grid_vars(location)[0].shape
    d_shp = len(n_depth) if n_depth is not None else None
    t_shp = len(o_var.time) if o_var.time is not None else None
    n_shape = xy_shp
    if d_shp:
        n_shape = (d_shp,) + n_shape
    if t_shp:
        n_shape = (t_shp,) + n_shape

    n_data = np.empty(n_shape)
    location_shp = grid.node_lon.shape

    pts = np.zeros((dest_points.shape[0], 3))
    pts[:, 0:2] = dest_points
    if o_var.time is not None:
        for t_idx, t in enumerate(o_var.time.data):
            if n_depth is not None and issubclass(n_depth.__class__, S_Depth):
                transect = o_var.depth.get_transect(o_var.grid.nodes.reshape(-1,2), t, data_shape=(len(n_depth),)).T
                for lev_idx, lev_data in enumerate(transect):
                    lev = Variable(name='level{0}'.format(lev_idx),
                                   data=lev_data.reshape(o_var.grid.node_lon.shape),
                                   grid=o_var.grid)
                    zs = lev.at(pts, t)
                    pts[:, 2] = zs[:,0]
                    n_data[t_idx, lev_idx] = o_var.at(pts, t).reshape(location_shp)
            else:
                n_data[t_idx] = o_var.at(pts, t).reshape(location_shp)
    else:
        n_data = o_var.at(pts, None).reshape(location_shp)

    n_var = Variable(name='regridded {0}'.format(o_var.name),
                     grid=grid,
                     time=o_var.time,
                     depth=n_depth,
                     data=n_data,
                     units=o_var.units)
    return n_var


def _regrid_s_depth(grid, o_depth):
    """
    Creates a new S_Depth object from an existing one that works on a new grid.
    """
    o_bathy = o_depth.bathymetry
    o_zeta = o_depth.zeta
    n_bathy = regrid_variable(grid, o_bathy)
    n_zeta = regrid_variable(grid, o_zeta)
    n_depth = o_depth.__class__(time=o_depth.time,
                      grid=grid,
                      bathymetry=n_bathy,
                      zeta=n_zeta,
                      terms={'Cs_w': o_depth.Cs_w,
                             'Cs_r': o_depth.Cs_r,
                             's_w': o_depth.s_w,
                             's_rho': o_depth.s_rho,
                             'hc': o_depth.hc})
    return n_depth


def _reorganize_spatial_data(points):
    """
    Provides a version of the points data organized for use in internal gridded
    algorithms. The data should be organized as an array with N rows, with
    with each row representing the coordinates of one point. Each row should
    at least have a length of two. The first column is ALWAYS considered to be
    longitude (or x_axis) and the second is latitude (or y_axis). The third is
    depth (or z_axis).

    In particular, this function is meant to address situations where the user
    provides spatial data in a non-numpy container, or with a different shape
    than expected. For example:

    single_pt = np.array([[(4,),(5,)]])
    array([[4],
           [5]])

    The assumed interpretation is that this represents one point with lon = 4,
    lat = 5. The reorganization for internal use would be:

    array([[4, 5]])

    In the situation where the shape of the input data is shaped between 2x2
    and 3x3, no changes are made, because it is impossible to tell which
    'orientation' is more 'correct'. In this case, the data will be used as is.

    Since the user may organize their spatial data in a number of ways, this
    function should be used to standardize the format so math can be done
    consistently. It is also worth using the _align_results_to_spatial_data function
    in case it is appropriate to reformat calculation results to be more
    like the spatial data input.

    array([[lon0, lat0, depth0],
           [lon1, lat1, depth1],
           ...
           [lonN, latN, depthN]])
    """
    if points is None:
        return None
    points_arr = np.array(points).squeeze()
    if points_arr.dtype in (np.object_, np.bytes_, np.bool_):
        raise TypeError('points data does not convert to a numeric array type')
    shp = points_arr.shape
    # special cases
    if len(shp) == 1:
        #two valid cases: [lon, lat] and [lon, lat, depth]. all others = error
        if shp == (2,) or shp == (3,):
            return points_arr.reshape(1,shp[0])
        else:
            raise ValueError('Only 2 or 3 elements (lon, lat) or\
                             (lon, lat_depth) is allowed when using 1D data')
    if shp in ((2,2),(2,3),(3,2),(3,3)):
        #can't do anything here because it's 100% ambiguous.
        return points_arr
    else:
        '''
        Make sure at least one of the dimensions is length 2 or 3. If it is
        dimension 0, then reshape, otherwise do nothing. If both dimensions are
        longer than 3, this is an error.
        '''
        if shp[0] > 3 and shp[1] > 3:
            raise ValueError('Too many coordinate dimensions in input array')
        if shp[0] > shp[1]:
            #'correct' shape
            return points_arr
        else:
            return points_arr.T
        pass



def _align_results_to_spatial_data(result, points):
    """
    Takes the results of a calculation and reorganizes it so it's dimensions
    are in line with the original spatial data. This function should only be
    used when the output of a calculation has a one-to-one correlation with
    some input spatial data. For example:
                      #[x,y,z]
    points = np.array([[1,2,3],
                       [4,5,6],
                       [6,7,8],
                       [9,10,11)

                       [x,y]
    result = np.array([[1,1],
                       [2,2],
                       [3,3],
                       [4,4]])

    In this case, result is 'aligned' with the input points (there is 1 row in
    result for each [x,y,z] in points. This format of one row output per input
    point should be consistent for vectorized calculations in this library

    points = np.array([[1,4,6,9],   #x
                       [2,5,7,10],  #y
                       [3,6,8,11]]) #z

    However, this case is not aligned with result from above. Therefore we
    should re-organize the result

    result = np.array([[1,2,3,4],   #x
                       [1,2,3,4]])  #y

    This function is not applied when input points or results shape are in the
    domain between 2x2 and 3x3. If the result cannot be aligned, then the
    result is returned unchanged. Note that this is an internal function
    and should be used specifically when implementing interfaces for functions
    that are expected to provide strictly vectorized results.
    (N results per N input points)
    """
    if points is None :
        return result
    points_arr = np.array(points).squeeze()
    shp = points_arr.shape
    #special cases
    if len(shp) == 1:
        #two valid cases: [lon, lat] and [lon, lat, depth]. all others = error
        if shp == (2,) or shp == (3,):
            return result
        else:
            raise ValueError('Only 2 or 3 elements (lon, lat) or\
                             (lon, lat_depth) is allowed when using 1D data')
    elif shp in ((2,2),(2,3),(3,2),(3,3)):
        return result
    else:
        if shp[0] > 3 and shp[1] > 3:
            raise ValueError('Too many coordinate dimensions in points array')
        elif shp[0] < shp[1] and result.shape[0] == shp[1]:
            return result.T
        else:
            if shp[0] == result.shape[0] and len(result.shape) == 1:
                return result[:, None] #enforce (N,1) shape
            else:
                return result


def isarraylike(obj):
    """
    tests if obj acts enough like an array to be used in gridded.

    This should catch netCDF4 variables and numpy arrays, at least, etc.

    Note: these won't check if the attributes required actually work right.
    """
    for attr in must_have:
        if not hasattr(obj, attr):
            return False
    return True


def asarraylike(obj):
    """
    If it satisfies the requirements of pyugrid the object is returned as is.
    If not, then numpy's array() will be called on it.

    :param obj: The object to check if it's like an array

    """

    return obj if isarraylike(obj) else np.array(obj)


def isstring(obj):
    """
    py2/3 compatble way to test for a string
    """
    try:
        return isinstance(obj, basestring)
    except NameError:
        return isinstance(obj, str)


def get_dataset(ncfile, dataset=None):
    """
    Utility to create a netCDF4 Dataset from a filename, list of filenames,
    or just pass it through if it's already a netCDF4.Dataset

    if dataset is not None, it should be a valid netCDF4 Dataset object,
    and it will simply be returned

    fixme: why can you pass in a dataset specifically ???
    """
    if dataset is not None:
        return dataset
    if isinstance(ncfile, (nc4.Dataset, nc4.MFDataset)):
        return ncfile
    try:
        ncfile = os.fspath(ncfile)
    except TypeError:
        pass  # not a string of PathLike -- could be a list or invalid ...

    if isinstance(ncfile, str):  # single path
        return nc4.Dataset(ncfile)
    elif isinstance(ncfile, Iterable) and len(ncfile) == 1:
        return nc4.Dataset(ncfile[0])
    else:
        return nc4.MFDataset(ncfile)

def parse_filename_dataset_args(filename=None,
                                dataset=None,
                                data_file=None,
                                grid_file=None,):
    """A perinnial problem is that we need to be able to accept a variety of
    filename/dataset arguments. This function will return a tuple of 'ds' and 'dg'
    where ds is the netCDF4.Dataset for the data and dg is the netCDF4.Dataset for the grid. If such
    datasets are from the same file, it will return the same object for both.
    
    :param filename: str or pathlike
    :param dataset: netCDF4.Dataset
    :param data_file: str, pathlike, or netCDF4.Dataset
    :param grid_file: str, pathlike, or netCDF4.Dataset
    """
    ds = None
    dg = None
    if filename is not None:
        try:
            filename = os.fspath(filename)
        except TypeError:
            pass
        data_file = grid_file = filename
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
    
    return ds, dg

def get_writable_dataset(ncfile, format="netcdf4"):
    """
    Utility to create a writable netCDF4 Dataset from a filename, list of filenames,
    or just pass it through if it's already a netCDF4.Dataset

    if dataset is not None, it should be a valid netCDF4 Dataset object,
    and it will simply be returned
    """
    if isinstance(ncfile, nc4.Dataset):
        # fixme: check for writable
        return ncfile
    elif isstring(ncfile):  # Fixme: should be pathlike...
        print("filename is:", ncfile)
        return nc4.Dataset(ncfile,
                           mode="w",
                           clobber=True,
                           format="NETCDF4")
    else:
        raise ValueError("Must be a string path or a netcdf4 Dataset")


def get_dataset_attrs(ds):
    """
    get all the attributes of the dataset as a single dict

    :param ds: an open netCDF4.Dataset
    """
    return {name: ds.getncattr(name) for name in ds.ncattrs()}

def search_netcdf_vars(cls=None, ds=None, dg=None):
    """
    Given a class with a .default_names and .cf_names attributes, search a datafile 
    netCDF4.Dataset and a possible grid netCDF4.Dataset for variables that are sought by
    the class
    """
    vn_search = search_dataset_for_variables_by_varname(ds, cls.default_names)
    ds_search = search_dataset_for_variables_by_longname(ds, cls.cf_names)
    found_vars = merge_var_search_dicts(ds_search, vn_search)
    if ds != dg:
        dg_vn_search = search_dataset_for_variables_by_varname(dg, cls.default_names)
        dg_ln_search = search_dataset_for_variables_by_longname(dg, cls.cf_names)
        dg_search = merge_var_search_dicts(dg_ln_search, dg_vn_search)
        found_vars = merge_var_search_dicts(found_vars, dg_search)
    return found_vars
    
def can_create_class(cls, ds=None, dg=None):
    found_vars = search_netcdf_vars(cls, ds, dg)
    # all variables must be found (no None values)
    return not (None in found_vars.values())

def varnames_merge(cls, inc_varnames=None):
    """
    Helper function to support the `varnames` argument pattern.

    `varnames` is a keyword used to specify an association between a desired subcomponent
    of an object and a desired data source name. It is meant to be a mechanism by which 
    custom digestion of a file can be specified.
    """

def search_dataset_for_any_long_name(ds, names):
    """
    Searches a netCDF4.Dataset for any netCDF4.Variable that satisfies one of the search terms.
    :param name: list of strings or str -> list dictionary

    :returns: boolean
    """
    if isinstance(names, Mapping):
        names = sum(list(names.values()), [])
            
    for n in names:
        t1 = ds.get_variables_by_attributes(long_name=n)
        t2 = ds.get_variables_by_attributes(standard_name=n)
        if t1 or t2:
            return t1+t2

def search_dataset_for_variables_by_longname(ds, possible_names):
    """
    For each longname list in possible_names, search the Dataset

    :param ds: Dataset
    :type ds: netCDF4.Dataset
    :param possible_names: str -> list dictionary
    :type possible_names: dict

    :returns: str -> netCDF4.Variable dictionary
    """
    rtv = {}
    for k, v in possible_names.items():
        for query in v:
            t1 = ds.get_variables_by_attributes(long_name=query)
            t2 = ds.get_variables_by_attributes(standard_name=query)
            if t1 or t2:
                rtv[k] = (t1+t2)[0]
                break
        if k not in rtv:
            rtv[k] = None
    return rtv

def search_dataset_for_variables_by_varname(ds, possible_names):
    """
    For each varname list in possible_names, search the Dataset

    :param ds: Dataset
    :type ds: netCDF4.Dataset
    :param possible_names: str -> list dictionary
    :type possible_names: dict

    :returns: str -> netCDF4.Variable (or None if not found) dictionary
    """
    rtv = {}
    for k, v in possible_names.items():
        for query in v:
            if query in ds.variables:
                rtv[k] = ds.variables[query]
                break
        if k not in rtv:
            rtv[k] = None
    return rtv

def merge_var_search_dicts(d1, d2):
    """
    Values in D1 take precedence over D2 if they are not None
    :param d1: str -> netCDF4.Variable dict
    :type d1: dict
    :param d1: str -> netCDF4.Variable dict
    :type d2: dict

    :returns: str -> netCDF4.Variable dict
    """
    result = d1.copy()
    for k, v in d2.items():
        if k not in d1:
            result[k] = v
        else:
            if result[k] is None:
                result[k] = v
    return result
