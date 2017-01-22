"""
Miscellaneous util functions.

"""

from __future__ import (absolute_import, division, print_function)

import numpy as np


epsilon = 1.e-5


def point_in_tri(face_points, point, return_weights=False):
    """
    Calculates whether point is internal/external
    to element by comparing summed area of sub triangles with area of triangle
    element.

    """
    sub_tri_areas = np.zeros(3)
    sub_tri_areas[0] = _signed_area_tri(np.vstack((face_points[(0, 1), :],
                                                   point)))
    sub_tri_areas[1] = _signed_area_tri(np.vstack((face_points[(1, 2), :],
                                                   point)))
    sub_tri_areas[2] = _signed_area_tri(np.vstack((face_points[(0, 2), :],
                                                   point)))
    tri_area = _signed_area_tri(face_points)

    if abs(np.abs(sub_tri_areas).sum()-tri_area)/tri_area <= epsilon:
        if return_weights:
            raise NotImplementedError
            # weights = sub_tri_areas/tri_area
            # weights[1] = max(0., min(1., weights[1]))
            # weights[2] = max(0., min(1., weights[2]))
            # if (weights[0]+weights[1]>1):
            #     weights[2] = 0
            #     weights[1] = 1-weights[0]
            # else:
            #     weights[2] = 1-weights[0]-weights[1]
            #
            # return weights
        else:
            return True
    return False


def _signed_area_tri(points):
    """
    points : the coordinates of the triangle vertices -- (3x2) float array
    returns signed area of the triangle.

    """

    x1, y1 = points[0]
    x2, y2 = points[1]
    x3, y3 = points[2]

    return(((x1-x3)*(y2-y3)-(x2-x3)*(y1-y3))/2)


must_have = ['dtype', 'shape', 'ndim','__len__', '__getitem__', '__getattribute__']
def isarraylike(obj):
    """
    tests if obj acts enough like an array to be used in pyugrid.

    This should catch netCDF4 variables, etc.

    Note: these won't check if the attributes required actually work right.
    """
    for attr in must_have:
        if not hasattr(obj, attr):

            return False

    return True

def asarraylike(obj):
    """
    If it satisfies the requirements of pyugrid the object is returned as is. If not, then numpy's
    array() will be called on it.

    :param obj: The object to check if it's like an array

    """

    return obj if isarraylike(obj) else np.array(obj)


