#!/usr/bin/env python

"""
assorted utility functions needed by gridded
"""

import collections
import numpy as np
import netCDF4 as nc4

must_have = ['dtype', 'shape', 'ndim', '__len__', '__getitem__', '__getattribute__']


def _reorganize_spatial_data(points):
    """
    Provides a version of the data organized for use in internal gridded
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
    consistently. It is also worth using the _spatial_data_metadata function
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
    if points_arr.dtype in (np.object_, np.string_, np.bool_):
        raise TypeError('points data does not convert to a numeric array type')
    shp = points_arr.shape
    #special cases
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
    py2/3 compaitlbie wayto test for a string
    """
    try:
        return isinstance(obj, basestring)
    except:
        return isinstance(obj, str)


def get_dataset(ncfile, dataset=None):
    """
    Utility to create a netCDF4 Dataset from a filename, list of filenames,
    or just pass it through if it's already a netCDF4.Dataset

    if dataset is not None, it should be a valid netCDF4 Dataset object,
    and it will simiply be returned
    """
    if dataset is not None:
        return dataset
    if isinstance(ncfile, nc4.Dataset):
        return ncfile
    elif isinstance(ncfile, collections.Iterable) and len(ncfile) == 1:
        return nc4.Dataset(ncfile[0])
    elif isstring(ncfile):
        return nc4.Dataset(ncfile)
    else:
        return nc4.MFDataset(ncfile)
