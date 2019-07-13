#!/usr/bin/env python

"""
Tests for finding a given UVar by standard name.

FIXME: This could use some more testing and cleaning up.
No need for all this reading and writing; that is tested elsewhere.

"""

from __future__ import (absolute_import, division, print_function)

import os

import pytest
import numpy as np

from gridded.pyugrid.ugrid import UGrid
from gridded.pyugrid.uvar import UVar


from utilities import chdir, two_triangles, twenty_one_triangles

test_files = os.path.join(os.path.dirname(__file__), 'files')


@pytest.fixture
def two_triangles_with_depths():
    grid = two_triangles()

    depths = UVar('depth', location='node', data=[1.0, 2.0, 3.0, 4.0])
    depths.attributes['units'] = 'unknown'
    depths.attributes['standard_name'] = 'sea_floor_depth_below_geoid'
    depths.attributes['positive'] = 'down'
    grid.add_data(depths)

    return grid


@pytest.fixture
def twenty_one_triangles_with_depths():
    """Returns a basic triangle grid with 21 triangles, a hole and a tail."""
    grid = twenty_one_triangles()

    depths = UVar('depth', location='node', data=list(range(1, 21)))
    depths.attributes['units'] = 'unknown'
    depths.attributes['standard_name'] = 'sea_floor_depth_below_geoid'
    depths.attributes['positive'] = 'down'
    grid.add_data(depths)

    return grid


def find_depths(grid):
    found = grid.find_uvars('sea_floor_depth_below_geoid')
    if found:
        return found.pop()
    return None


def test_no_std_name():
    """
    Tests to make sure it doesn't crash if a `UVar` does not have a
    `standard_name`.

    """
    grid = two_triangles_with_depths()

    junk = UVar('junk', location='node', data=[1.0, 2.0, 3.0, 4.0])
    junk.attributes['units'] = 'unknown'
    grid.add_data(junk)

    depths = find_depths(grid)

    assert depths.name == 'depth'


def test_two_triangles():
    grid = two_triangles_with_depths()

    fname = '2_triangles.nc'
    with chdir(test_files):
        grid.save_as_netcdf(fname)
        ug = UGrid.from_ncfile(fname, load_data=True)
        os.remove(fname)

    assert ug.nodes.shape == (4, 2)
    assert ug.nodes.shape == grid.nodes.shape

    # FIXME: Not ideal to pull specific values out, but how else to test?
    assert np.array_equal(ug.nodes[0, :], (0.1, 0.1))
    assert np.array_equal(ug.nodes[-1, :], (3.1, 2.1))
    assert np.array_equal(ug.nodes, grid.nodes)

    depths = find_depths(ug)
    assert depths.data.shape == (4,)
    assert depths.data[0] == 1
    assert depths.attributes['units'] == 'unknown'


def test_21_triangles():
    grid = twenty_one_triangles_with_depths()

    fname = '21_triangles.nc'
    with chdir(test_files):
        grid.save_as_netcdf(fname)
        ug = UGrid.from_ncfile(fname, load_data=True)
        os.remove(fname)

    assert ug.nodes.shape == grid.nodes.shape

    # FIXME: Not ideal to pull specific values out, but how else to test?
    assert np.array_equal(ug.nodes, grid.nodes)

    depths = find_depths(ug)
    assert depths.data.shape == (20,)
    assert depths.data[0] == 1
    assert depths.attributes['units'] == 'unknown'


def test_two_triangles_without_faces():
    grid = two_triangles_with_depths()
    grid.faces = None

    fname = '2_triangles_without_faces.nc'
    with chdir(test_files):
        grid.save_as_netcdf(fname)
        ug = UGrid.from_ncfile(fname, load_data=True)
        os.remove(fname)

    assert ug.nodes.shape == (4, 2)
    assert ug.nodes.shape == grid.nodes.shape

    # FIXME: Not ideal to pull specific values out, but how else to test?
    assert np.array_equal(ug.nodes[0, :], (0.1, 0.1))
    assert np.array_equal(ug.nodes[-1, :], (3.1, 2.1))
    assert np.array_equal(ug.nodes, grid.nodes)

    assert ug.faces is None

    assert ug.edges.shape == grid.edges.shape
    assert np.array_equal(ug.edges[0, :], (0, 1))
    assert np.array_equal(ug.edges[3, :], (2, 0))

    depths = find_depths(ug)
    assert depths.data.shape == (4,)
    assert depths.data[0] == 1
    assert depths.attributes['units'] == 'unknown'


def test_two_triangles_without_edges():
    grid = two_triangles_with_depths()
    # This will set the _edges to None, but it will be rebuild
    grid.edges = None

    fname = '2_triangles_without_edges.nc'
    with chdir(test_files):
        grid.save_as_netcdf(fname)
        ug = UGrid.from_ncfile(fname, load_data=True)
        os.remove(fname)

    assert ug.nodes.shape == (4, 2)
    assert ug.nodes.shape == grid.nodes.shape

    # FIXME: Not ideal to pull specific values out, but how else to test?
    assert np.array_equal(ug.nodes[0, :], (0.1, 0.1))
    assert np.array_equal(ug.nodes[-1, :], (3.1, 2.1))
    assert np.array_equal(ug.nodes, grid.nodes)

    assert ug.faces.shape == grid.faces.shape

    # the edges rebuild from faces
    assert ug._edges is None
    ug.build_edges()
    assert ug.edges is not None

    depths = find_depths(ug)
    assert depths.data.shape == (4,)
    assert depths.data[0] == 1
    assert depths.attributes['units'] == 'unknown'


if __name__ == '__main__':
    test_two_triangles()
    test_21_triangles()
    test_two_triangles_without_faces()
    test_two_triangles_without_edges()
