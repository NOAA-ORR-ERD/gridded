"""
Created on Apr 3, 2015

@author: ayan

"""


import pytest
import numpy as np

from gridded.pysgrid.processing_2d import avg_to_cell_center, rotate_vectors, vector_sum


def test_vector_sum():
    x_vector = np.array([3, 5, 9, 11])
    y_vector = np.array([4, 12, 40, 60])
    sum_result = vector_sum(x_vector, y_vector)
    expected = np.array([5, 13, 41, 61])
    np.testing.assert_almost_equal(sum_result, expected)


@pytest.fixture
def rotate_vectors_data():
    x = np.array([3, 5, 9, 11])
    y = np.array([4, 12, 40, 60])
    angles_simple = np.array([0, np.pi / 2, 0, np.pi / 2])
    angles_complex = np.array([np.pi / 6, np.pi / 5,
                               np.pi / 4, np.pi / 3])
    return x, y, angles_simple, angles_complex


def test_vector_rotation_simple(rotate_vectors_data):
    x, y, angles_simple, angles_complex = rotate_vectors_data
    rotated_x, rotated_y = rotate_vectors(x, y, angles_simple)

    expected_x = np.array([3, -12, 9, -60])
    expected_y = np.array([4, 5, 40, 11])

    np.testing.assert_almost_equal(rotated_x, expected_x, decimal=3)
    np.testing.assert_almost_equal(rotated_y, expected_y, decimal=3)


def test_vector_rotation_complex(rotate_vectors_data):
    x, y, angles_simple, angles_complex = rotate_vectors_data
    rotated_x, rotated_y = rotate_vectors(x, y, angles_complex)
    expected_x = np.array([0.5981, -3.0083, -21.9203, -46.4615])
    expected_y = np.array([4.9641, 12.6471, 34.6482, 39.5263])
    np.testing.assert_almost_equal(rotated_x, expected_x, decimal=3)
    np.testing.assert_almost_equal(rotated_y, expected_y, decimal=3)


@pytest.fixture
def avg_center_data():
    return np.array([[4, 5, 9, 10], [8, 39, 41, 20], [5, 29, 18, 71]])


def test_no_transpose(avg_center_data):
    data = avg_center_data
    avg_result = avg_to_cell_center(data, 1)
    expected = np.array([[4.5, 7, 9.5],
                         [23.5, 40, 30.5],
                         [17, 23.5, 44.5]])
    np.testing.assert_almost_equal(avg_result, expected, decimal=3)


def test_with_transpose(avg_center_data):
    data = avg_center_data
    avg_result = avg_to_cell_center(data, 0)
    expected = np.array([[6, 22, 25, 15], [6.5, 34, 29.5, 45.5]])
    np.testing.assert_almost_equal(avg_result, expected, decimal=3)
