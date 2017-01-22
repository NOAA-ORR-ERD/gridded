#!/usr/bin/env python

"""
assorted utility functions needed by gridded
"""

import numpy as np
import netCDF4 as nc4

must_have = ['dtype', 'shape', 'ndim', '__len__', '__getitem__', '__getattribute__']


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
    If it satisfies the requirements of pyugrid the object is returned as is. If not,
    then numpy's array() will be called on it.

    :param obj: The object to check if it's like an array

    """

    return obj if isarraylike(obj) else np.array(obj)


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
    elif isinstance(ncfile, basestring):
        return nc4.Dataset(ncfile)
    else:
        return nc4.MFDataset(ncfile)

