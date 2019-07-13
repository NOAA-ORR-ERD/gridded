#!/usr/bin/env python

"""
Tests for writing and reading UGRID compliant netCDF.

"""

from __future__ import (absolute_import, division, print_function)

import os
import numpy as np

import pytest

from gridded.pyugrid.ugrid import UGrid
from gridded.pyugrid.uvar import UVar

from utilities import chdir, two_triangles

pytestmark = pytest.mark.skipif(True, reason="gridded does not support UVars anymore")


test_files = os.path.join(os.path.dirname(__file__), 'files')


def test_with_faces(two_triangles):
    """
    Test with faces, edges, but no `face_coordinates` or `edge_coordinates`.

    """

    expected = two_triangles

    fname = '2_triangles.nc'
    with chdir(test_files):
        expected.save_as_netcdf(fname)
        grid = UGrid.from_ncfile(fname)
        os.remove(fname)

    assert np.array_equal(expected.nodes, grid.nodes)
    assert np.array_equal(expected.faces, grid.faces)
    assert np.array_equal(expected.edges, grid.edges)


def test_without_faces(two_triangles):
    expected = two_triangles
    del expected.faces
    assert expected.faces is None

    fname = '2_triangles.nc'
    with chdir(test_files):
        expected.save_as_netcdf(fname)
        grid = UGrid.from_ncfile(fname)
        os.remove(fname)

    assert grid.faces is None
    assert np.array_equal(expected.faces, grid.faces)
    assert np.array_equal(expected.edges, grid.edges)


def test_with_just_nodes_and_depths(two_triangles):
    expected = two_triangles
    del expected.faces
    del expected.edges

    depth = UVar('depth',
                 'node',
                 np.array([1.0, 2.0, 3.0, 4.0]),
                 {'units': 'm',
                  'positive': 'down',
                  'standard_name': 'sea_floor_depth_below_geoid'})
    expected.add_data(depth)

    fname = '2_triangles_depth.nc'
    with chdir(test_files):
        expected.save_as_netcdf(fname)
        grid = UGrid.from_ncfile(fname, load_data=True)
        os.remove(fname)

    assert grid.faces is None
    assert grid.edges is None
    assert np.array_equal(expected.nodes, grid.nodes)

    assert np.array_equal(expected.data['depth'].data, grid.data['depth'].data)
    assert expected.data['depth'].attributes == grid.data['depth'].attributes


if __name__ == "__main__":
    test_with_faces()
    test_without_faces()
