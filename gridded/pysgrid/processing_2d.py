'''
Created on Apr 2, 2015

@author: ayan
'''


import numpy as np
import numpy.ma as ma


def vector_sum(x_arr, y_arr):
    """
    Calculate the vector sum of arrays of
    x and y vectors.

    :param x_arr: array of x-directed vectors
    :type x_arr: numpy.array
    :param y_arr: array of y-directed vectors
    :type y_arr: numpy.array
    :return: array of vector sums
    :rtype: numpy.array

    """
    return ma.sqrt(x_arr**2 + y_arr**2)


def rotate_vectors(x_arr, y_arr, angle_arr):
    """
    Given x and y vectors in a projected coordinate
    system, rotate them by angles into a different
    coordinate system.

    All arrays must have the same dimensions.

    :param x_arr: array of x-directed vectors
    :type x_arr: numpy.array
    :param y_arr: array of y-directed vectors
    :type y_arr: numpy.array
    :param angle_arr: array of angles in radians
    :type angle_arr: numpy.array
    :return: x and y arrays of rotated vectors
    :rtype: tuple

    """
    x_rot = x_arr*np.cos(angle_arr) - y_arr*np.sin(angle_arr)
    y_rot = x_arr*np.sin(angle_arr) + y_arr*np.cos(angle_arr)
    return x_rot, y_rot


def avg_to_cell_center(data_array, avg_dim):
    """
    Given a two-dimensional numpy.array, average
    adjacent row values (avg_dim=1) or adjacent
    column values (avg_dim=0) to the grid cell
    center.

    :param data_array: 2-dimensional data
    :type data_array: numpy.array
    :param int avg_dim: integer specify array axis to be averaged
    :return: averages
    :rtype: numpy.array

    """
    if avg_dim == 0:
        da = np.transpose(data_array)
    else:
        da = data_array
    da_trim_low = da[:, 1:]
    da_trim_high = da[:, :-1]
    da_avg_raw = 0.5 * (da_trim_low + da_trim_high)
    if avg_dim == 0:
        da_avg = np.transpose(da_avg_raw)
    else:
        da_avg = da_avg_raw
    return da_avg
