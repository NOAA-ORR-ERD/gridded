"""
Test SGrid Variable ROMS.

Created on Apr 15, 2015

@author: ayan

"""

from __future__ import (absolute_import, division, print_function)

import pytest

import numpy as np

from gridded.pysgrid.sgrid import SGrid
from gridded.pysgrid.utils import GridPadding
from gridded.pysgrid.variables import SGridVariable
from .write_nc_test_files import roms_sgrid


@pytest.fixture
def sgrid_variable_roms(roms_sgrid):
    face_padding = [GridPadding(mesh_topology_var=u'grid',
                                face_dim=u'xi_rho',
                                node_dim=u'xi_psi',
                                padding=u'both'),
                    GridPadding(mesh_topology_var=u'grid',
                                face_dim=u'eta_rho',
                                node_dim=u'eta_psi',
                                padding=u'both')]
    return dict(
        sgrid=SGrid(face_padding=face_padding,
                    node_dimensions='xi_psi eta_psi'),
        test_var_1=roms_sgrid.variables['u'],
        test_var_2=roms_sgrid.variables['zeta'],
        test_var_3=roms_sgrid.variables['salt'],
        test_var_4=roms_sgrid.variables['fake_u'])


def test_create_sgrid_variable_object(sgrid_variable_roms):
    sgrid_var = SGridVariable.create_variable(
        sgrid_variable_roms['test_var_1'],
        sgrid_variable_roms['sgrid']
    )
    assert isinstance(sgrid_var, SGridVariable)


def test_attributes_with_grid(sgrid_variable_roms):
    sgrid_var = SGridVariable.create_variable(
        sgrid_variable_roms['test_var_1'],
        sgrid_variable_roms['sgrid']
    )
    sgrid_var_name = sgrid_var.variable
    sgrid_var_name_expected = 'u'
    sgrid_var_dim = sgrid_var.dimensions
    sgrid_var_dim_expected = ('time', 's_rho', 'eta_u', 'xi_u')
    sgrid_var_grid = sgrid_var.grid
    sgrid_var_grid_expected = 'some grid'
    sgrid_var_location = sgrid_var.location
    sgrid_var_location_expected = 'edge1'
    sgrid_var_dtype = sgrid_var.dtype
    x_axis = sgrid_var.x_axis
    x_axis_expected = 'xi_u'
    y_axis = sgrid_var.y_axis
    y_axis_expected = 'eta_u'
    z_axis = sgrid_var.z_axis
    standard_name = sgrid_var.standard_name
    expected_standard_name = 'sea_water_x_velocity'
    coordinates = sgrid_var.coordinates
    expected_coordinates = ('time', 's_rho', 'lat_u', 'lon_u')

    assert sgrid_var_name == sgrid_var_name_expected
    assert sgrid_var_dim == sgrid_var_dim_expected
    assert sgrid_var_grid == sgrid_var_grid_expected
    assert sgrid_var_location == sgrid_var_location_expected
    assert sgrid_var_dtype == np.dtype('float32')
    assert x_axis == x_axis_expected
    assert y_axis == y_axis_expected
    assert z_axis is None
    assert standard_name == expected_standard_name
    assert coordinates == expected_coordinates


def test_face_location_inference(sgrid_variable_roms):
    sgrid_var = SGridVariable.create_variable(
        sgrid_variable_roms['test_var_3'],
        sgrid_variable_roms['sgrid']
    )
    sgrid_var_location = sgrid_var.location
    expected_location = 'face'
    assert sgrid_var_location == expected_location


def test_edge_location_inference(sgrid_variable_roms):
    sgrid_var = SGridVariable.create_variable(
        sgrid_variable_roms['test_var_4'],
        sgrid_variable_roms['sgrid']
    )
    sgrid_var_location = sgrid_var.location
    # Representative ROMS sgrid.
    # None is expected since `edge1` and `edge2` attributes are not defined.
    assert sgrid_var_location is None


def test_edge_location_inference_with_defined_edges(sgrid_variable_roms):
    sgrid_variable_roms['sgrid'].edge1_padding = [
        GridPadding(mesh_topology_var=u'grid',
                    face_dim=u'eta_u',
                    node_dim=u'eta_psi',
                    padding=u'both')
    ]
    sgrid_variable_roms['sgrid'].edge2_padding = [
        GridPadding(mesh_topology_var=u'grid',
                    face_dim=u'xi_v',
                    node_dim=u'xi_psi',
                    padding=u'both')
    ]
    sgrid_var = SGridVariable.create_variable(
        sgrid_variable_roms['test_var_4'], sgrid_variable_roms['sgrid'])
    sgrid_var_location = sgrid_var.location
    expected_location = 'edge1'
    assert sgrid_var_location == expected_location


def test_attributes_with_location(sgrid_variable_roms):
    sgrid_var = SGridVariable.create_variable(
        sgrid_variable_roms['test_var_2'], sgrid_variable_roms['sgrid'])
    sgrid_var_name = sgrid_var.variable
    sgrid_var_name_expected = 'zeta'
    sgrid_var_dim = sgrid_var.dimensions
    sgrid_var_grid = sgrid_var.grid
    sgrid_var_location = sgrid_var.location
    sgrid_var_location_expected = 'face'
    sgrid_var_dim_expected = ('time', 'eta_rho', 'xi_rho')
    sgrid_var_dtype = sgrid_var.dtype
    x_axis = sgrid_var.x_axis
    y_axis = sgrid_var.y_axis
    z_axis = sgrid_var.z_axis
    assert sgrid_var_name == sgrid_var_name_expected
    assert sgrid_var_dim == sgrid_var_dim_expected
    assert sgrid_var_grid is None
    assert sgrid_var_location == sgrid_var_location_expected
    assert sgrid_var_dtype == np.dtype('float32')
    assert x_axis is None
    assert y_axis is None
    assert z_axis is None


def test_vector_directions(sgrid_variable_roms):
    u_var = SGridVariable.create_variable(sgrid_variable_roms['test_var_1'],
                                          sgrid_variable_roms['sgrid'])
    u_vector_axis = u_var.vector_axis
    expected_u_axis = 'X'
    zeta_var = SGridVariable.create_variable(sgrid_variable_roms['test_var_2'],
                                             sgrid_variable_roms['sgrid'])
    zeta_axis = zeta_var.vector_axis
    assert u_vector_axis == expected_u_axis
    assert zeta_axis is None
