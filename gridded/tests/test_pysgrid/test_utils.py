"""
Created on Mar 23, 2015

@author: ayan

"""

from __future__ import (absolute_import, division, print_function)

import pytest

import numpy as np

from gridded.pysgrid.utils import (calculate_bearing,
                           calculate_angle_from_true_east,
                           check_element_equal,
                           does_intersection_exist,
                           pair_arrays,
                           )


@pytest.fixture
def intersection_data():
    a = (718, 903, 1029, 1701)
    b = (718, 828)
    c = (15, 20)
    return a, b, c


def test_intersect_exists():
    a, b, c = intersection_data()
    result = does_intersection_exist(a, b)
    assert result


def test_intersect_does_not_exist():
    a, b, c = intersection_data()
    result = does_intersection_exist(a, c)
    assert result is False


def test_pair_arrays():
    a1, a2, a3 = (1, 2), (3, 4), (5, 6)
    b1, b2, b3 = (10, 20), (30, 40), (50, 60)
    a = np.array([a1, a2, a3])
    b = np.array([b1, b2, b3])
    result = pair_arrays(a, b)
    expected = np.array([[(1, 10), (2, 20)],
                         [(3, 30), (4, 40)],
                         [(5, 50), (6, 60)]])
    np.testing.assert_almost_equal(result, expected, decimal=3)


@pytest.fixture
def check_element():
    """
    FIXME: These tests only check for a True return and not for correctness.

    """
    a = [7, 7, 7, 7]
    b = [7, 8, 9, 10]
    return a, b


def test_list_with_identical_elements():
    a, b = check_element()
    result = check_element_equal(a)
    assert result


def test_list_with_different_elements():
    a, b = check_element()
    result = check_element_equal(b)
    assert result is False


def test_bearing_calculation():
    points = np.array([(-93.51105439, 11.88846735),
                       (-93.46607342, 11.90917952)])
    point_1 = points[:-1, :]
    point_2 = points[1:, :]
    result = calculate_bearing(point_1, point_2)
    expected = 64.7947
    np.testing.assert_almost_equal(result, expected, decimal=3)


def test_angle_from_true_east_calculation():
    vertical_1 = np.array([[-122.41, 37.78],
                           [-122.33, 37.84],
                           [-122.22, 37.95]])
    vertical_2 = np.array([[-90.07, 29.95],
                           [-89.97, 29.84],
                           [-89.91, 29.76]])
    vertical_3 = np.array([[-89.40, 43.07],
                           [-89.49, 42.93],
                           [-89.35, 42.84]])
    vertical_4 = np.array([[-122.41, 37.78],
                           [-122.53, 37.84],
                           [-122.67, 37.95]])
    centers = np.array((vertical_1,
                        vertical_2,
                        vertical_3,
                        vertical_4))

    bearing_start_points = centers[:, :-1, :]
    bearing_end_points = centers[:, 1:, :]
    angle_from_true_east = calculate_angle_from_true_east(bearing_start_points, bearing_end_points)  # noqa
    expected_values = np.array([[0.7598, 0.9033, 0.9033],
                                [-0.903, -0.994, -0.994],
                                [-2.011, -0.719, -0.719],
                                [-3.706, -3.926, -3.926]])
    expected_shape = (4, 3)
    np.testing.assert_almost_equal(angle_from_true_east,
                                   expected_values, decimal=3)
    assert angle_from_true_east.shape == expected_shape
