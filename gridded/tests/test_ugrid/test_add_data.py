#!/usr/bin/env python

"""
Tests for adding a data attribute to a UGrid object.

"""


from __future__ import (absolute_import, division, print_function)

import numpy as np
import pytest

# from gridded.pyugrid import UVar

from utilities import two_triangles


pytestmark = pytest.mark.skipif(True, reason="gridded does not support UVars anymore")

#pytest.mark.skipif(True, "gridded does not support UVars anymore")


def test_add_all_data(two_triangles):
    """You should not be able add a data dict directly."""
    grid = two_triangles

    assert grid.data == {}

    with pytest.raises(AttributeError):
        grid.data = {'depth': UVar('depth',
                                   location='node',
                                   data=[1.0, 2.0, 3.0, 4.0])}


def test_add_node_data(two_triangles):
    grid = two_triangles

    # Create a UVar object for the depths:
    depths = UVar('depth', location='node', data=[1.0, 2.0, 3.0, 4.0])
    depths.attributes['units'] = 'm'
    depths.attributes['standard_name'] = 'sea_floor_depth'
    depths.attributes['positive'] = 'down'

    grid.add_data(depths)

    assert grid.data['depth'].name == 'depth'
    assert grid.data['depth'].attributes['units'] == 'm'
    assert np.array_equal(grid.data['depth'].data, [1.0, 2.0, 3.0, 4.0])


def test_add_node_data_wrong(two_triangles):
    """Too short an array."""

    grid = two_triangles

    # Create a UVar object for the depths:
    depths = UVar('depth', location='node', data=[1.0, 2.0, 3.0])

    with pytest.raises(ValueError):
        grid.add_data(depths)


def test_add_face_data(two_triangles):
    grid = two_triangles

    # Create a UVar object for velocity:
    u_vel = UVar('u', location='face', data=[1.0, 2.0])
    u_vel.attributes['units'] = 'm/s'
    u_vel.attributes['standard_name'] = 'eastward_sea_water_velocity'

    grid.add_data(u_vel)

    assert grid.data['u'].name == 'u'
    assert grid.data['u'].attributes['units'] == 'm/s'
    assert np.array_equal(grid.data['u'].data, [1.0, 2.0])


def test_add_face_data_wrong(two_triangles):
    """Too short an array."""

    grid = two_triangles

    # Create a UVar object for velocity:
    u_vel = UVar('u', location='face', data=[1.0])

    with pytest.raises(ValueError):
        grid.add_data(u_vel)


def test_add_edge_data(two_triangles):
    grid = two_triangles

    # Create a UVar object for velocity:
    bnds = UVar('bounds', location='edge', data=[0, 1, 0, 0, 1])
    bnds.attributes['standard_name'] = 'boundary type'

    grid.add_data(bnds)

    assert grid.data['bounds'].name == 'bounds'
    assert np.array_equal(grid.data['bounds'].data, [0, 1, 0, 0, 1])


def test_add_edge_data_wrong(two_triangles):
    """Too long an array."""
    grid = two_triangles

    # Create a UVar object for velocity:
    # a miss-matched set.
    bnds = UVar('bounds', location='edge', data=[0, 1, 0, 0, 1, 3, 3])
    with pytest.raises(ValueError):
        grid.add_data(bnds)


def test_add_boundary_data(two_triangles):
    grid = two_triangles

    # Add the boundary definitions:
    grid.boundaries = [(0, 1),
                       (0, 2),
                       (1, 3),
                       (2, 3)]

    # Create a UVar object for boundary conditions:
    bnds = UVar('bounds', location='boundary', data=[0, 1, 0, 0, 1])
    bnds.attributes['long_name'] = 'model boundary conditions'

    # Wrong size for data.
    with pytest.raises(ValueError):
        grid.add_data(bnds)

    # Correct data.
    bnds.data = [0, 1, 0, 0]
    grid.add_data(bnds)

    assert grid.data['bounds'].name == 'bounds'
    assert np.array_equal(grid.data['bounds'].data, [0, 1, 0, 0])
