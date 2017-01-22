#!/usr/bin/env python

"""
assorted utility functions needed by gridded
"""

import numpy as np

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
