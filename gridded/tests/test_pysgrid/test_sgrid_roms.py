"""
Created on Apr 7, 2015

@author: ayan

"""


import pytest
import numpy as np

from gridded.pysgrid.sgrid import SGrid, load_grid
from .write_nc_test_files import roms_sgrid


"""
Test SGrid ROMS.

"""


def test_load_from_dataset(roms_sgrid):
    sg_obj = load_grid(roms_sgrid)
    assert isinstance(sg_obj, SGrid)


@pytest.fixture
def sgrid(roms_sgrid):
    return load_grid(roms_sgrid)


def test_center_lon(sgrid):
    center_lon = sgrid.center_lon
    assert center_lon.shape == (4, 4)


def test_center_lat(sgrid):
    center_lat = sgrid.center_lat
    assert center_lat.shape == (4, 4)


def test_variables(sgrid):
    dataset_vars = sgrid.variables
    expected_vars = [u's_rho',
                     u's_w',
                     u'time',
                     u'xi_rho',
                     u'eta_rho',
                     u'xi_psi',
                     u'eta_psi',
                     u'xi_u',
                     u'eta_u',
                     u'xi_v',
                     u'eta_v',
                     u'grid',
                     u'u',
                     u'v',
                     u'fake_u',
                     u'lon_rho',
                     u'lat_rho',
                     u'lon_psi',
                     u'lat_psi',
                     u'lat_u',
                     u'lon_u',
                     u'lat_v',
                     u'lon_v',
                     u'salt',
                     u'zeta']
    assert len(dataset_vars) == len(expected_vars)
    assert dataset_vars == expected_vars


def test_grid_variables(sgrid):
    dataset_grid_variables = sgrid.grid_variables
    expected_grid_variables = [u'u', u'v', u'fake_u', u'salt']
    assert len(dataset_grid_variables) == len(expected_grid_variables)
    assert set(dataset_grid_variables) == set(expected_grid_variables)


def test_non_grid_variables(sgrid):
    dataset_non_grid_variables = sgrid.non_grid_variables
    expected_non_grid_variables = [u's_rho',
                                   u's_w',
                                   u'time',
                                   u'xi_rho',
                                   u'eta_rho',
                                   u'xi_psi',
                                   u'eta_psi',
                                   u'xi_u',
                                   u'eta_u',
                                   u'xi_v',
                                   u'eta_v',
                                   u'grid',
                                   u'lon_rho',
                                   u'lat_rho',
                                   u'lon_psi',
                                   u'lat_psi',
                                   u'lat_u',
                                   u'lon_u',
                                   u'lat_v',
                                   u'lon_v',
                                   u'zeta']
    assert len(dataset_non_grid_variables) == len(expected_non_grid_variables)
    assert set(dataset_non_grid_variables) == set(expected_non_grid_variables)


def test_variable_slicing(sgrid):
    u_center_slices = sgrid.u.center_slicing
    v_center_slices = sgrid.v.center_slicing
    u_center_expected = (np.s_[:], np.s_[:], np.s_[1:-1], np.s_[:])
    v_center_expected = (np.s_[:], np.s_[:], np.s_[:], np.s_[1:-1])
    assert u_center_slices == u_center_expected
    assert v_center_slices == v_center_expected


def test_grid_variable_average_axes(sgrid):
    uc_axis = sgrid.u.center_axis
    uc_axis_expected = 1
    un_axis = sgrid.u.node_axis
    un_axis_expected = 0
    lon_rho_c_axis = sgrid.lon_rho.center_axis
    lon_rho_n_axis = sgrid.lon_rho.node_axis
    assert uc_axis == uc_axis_expected
    assert un_axis == un_axis_expected
    assert lon_rho_c_axis is None
    assert lon_rho_n_axis is None


def test_optional_grid_attrs(sgrid):
    face_coordinates = sgrid.face_coordinates
    node_coordinates = sgrid.node_coordinates
    edge1_coordinates = sgrid.edge1_coordinates
    edge2_coordinates = sgrid.edge2_coordinates
    fc_expected = ('lon_rho', 'lat_rho')
    nc_expected = ('lon_psi', 'lat_psi')
    e1c_expected = ('lon_u', 'lat_u')
    e2c_expected = ('lon_v', 'lat_v')
    assert face_coordinates == fc_expected
    assert node_coordinates == nc_expected
    assert edge1_coordinates == e1c_expected
    assert edge2_coordinates == e2c_expected


def test_round_trip(sgrid, tmpdir):
    """
    TODO: add more "round-trip" tests.

    """
    fname = tmpdir.mkdir('files').join('deltares_roundtrip.nc')
    sgrid.save_as_netcdf(fname)
    result = load_grid(fname)
    assert isinstance(result, SGrid)
