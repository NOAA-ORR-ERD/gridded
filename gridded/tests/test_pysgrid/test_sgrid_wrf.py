"""
Created on Apr 7, 2015

@author: ayan

"""

from __future__ import (absolute_import, division, print_function)

import pytest
import numpy as np

from gridded.pysgrid.sgrid import SGrid, load_grid

from .write_nc_test_files import wrf_sgrid


"""
Test SGrid WRF Dataset.

"""


@pytest.fixture
def sgrid(wrf_sgrid):
    return load_grid(wrf_sgrid)


def test_topology_dimension(sgrid):
    topology_dim = sgrid.topology_dimension
    expected_dim = 2
    assert topology_dim == expected_dim


def test_variable_slicing_wrf(sgrid):
    u_slice = sgrid.U.center_slicing
    u_expected = (np.s_[:], np.s_[:], np.s_[:], np.s_[:])
    v_slice = sgrid.V.center_slicing
    v_expected = (np.s_[:], np.s_[:], np.s_[:], np.s_[:])
    assert u_slice == u_expected
    assert v_slice == v_expected


def test_variable_average_axes(sgrid):
    u_avg_axis = sgrid.U.center_axis
    u_axis_expected = 1
    v_avg_axis = sgrid.V.center_axis
    v_axis_expected = 0
    assert u_avg_axis == u_axis_expected
    assert v_avg_axis == v_axis_expected


def test_roundtrip(wrf_sgrid, tmpdir):
    """
    TODO: add more "round-trip" tests.

    Test SGrid Save No-Node Coordinates.
    """
    fname = tmpdir.mkdir('files').join('wrf_roundtrip.nc')
    sg_obj = load_grid(wrf_sgrid)
    sg_obj.save_as_netcdf(fname)


def test_sgrid(sgrid):
    assert isinstance(sgrid, SGrid)


def test_nodes(sgrid):
    node_lon = sgrid.node_lon
    node_lat = sgrid.node_lat
    assert node_lon is None
    assert node_lat is None


def test_node_coordinates(sgrid):
    node_coordinates = sgrid.node_coordinates
    assert node_coordinates is None


def test_node_dimensions(sgrid):
    node_dims = sgrid.node_dimensions
    expected = 'west_east_stag south_north_stag'
    assert node_dims == expected
