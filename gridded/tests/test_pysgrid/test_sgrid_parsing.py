"""
Created on Apr 7, 2015

@author: ayan

"""


import pytest

from gridded.pysgrid.read_netcdf import (parse_axes, parse_padding, parse_vector_axis)


def test_xyz_axis_parse():
    xyz = 'X: NMAX Y: MMAXZ Z: KMAX'
    result = parse_axes(xyz)
    expected = ('NMAX', 'MMAXZ', 'KMAX')
    assert result == expected


def test_xy_axis_parse():
    xy = 'X: xi_psi Y: eta_psi'
    result = parse_axes(xy)
    expected = ('xi_psi', 'eta_psi', None)
    assert result == expected


@pytest.fixture
def parse_pad():
    grid_topology = 'some_grid'
    pad_1 = 'xi_v: xi_psi (padding: high) eta_v: eta_psi'
    pad_2 = 'xi_rho: xi_psi (padding: both) eta_rho: eta_psi (padding: low)'
    pad_no = 'MMAXZ: MMAX NMAXZ: NMAX'
    return grid_topology, pad_1, pad_2, pad_no


def test_mesh_name(parse_pad):
    grid_topology, pad_1, pad_2, pad_no = parse_pad
    result = parse_padding(pad_1, grid_topology)
    mesh_topology = result[0].mesh_topology_var
    expected = 'some_grid'
    assert mesh_topology == expected


def test_two_padding_types(parse_pad):
    grid_topology, pad_1, pad_2, pad_no = parse_pad
    result = parse_padding(pad_2, grid_topology)
    expected_len = 2
    padding_datum_0 = result[0]
    padding_type = padding_datum_0.padding
    sub_dim = padding_datum_0.node_dim
    dim = padding_datum_0.face_dim
    expected_sub_dim = 'xi_psi'
    expected_padding_type = 'both'
    expected_dim = 'xi_rho'
    assert len(result) == expected_len
    assert padding_type == expected_padding_type
    assert sub_dim == expected_sub_dim
    assert dim == expected_dim


def test_one_padding_type(parse_pad):
    grid_topology, pad_1, pad_2, pad_no = parse_pad
    result = parse_padding(pad_1, grid_topology)
    expected_len = 1
    padding_datum_0 = result[0]
    padding_type = padding_datum_0.padding
    sub_dim = padding_datum_0.node_dim
    dim = padding_datum_0.face_dim
    expected_padding_type = 'high'
    expected_sub_dim = 'xi_psi'
    expected_dim = 'xi_v'
    assert len(result) == expected_len
    assert padding_type == expected_padding_type
    assert sub_dim == expected_sub_dim
    assert dim == expected_dim


def test_no_padding(parse_pad):
    grid_topology, pad_1, pad_2, pad_no = parse_pad
    with pytest.raises(ValueError):
        parse_padding(padding_str=pad_no,
                      mesh_topology_var=grid_topology)


def test_std_name_with_velocity_direction():
    standard_name_1 = 'sea_water_y_velocity'
    direction = parse_vector_axis(standard_name_1)
    expected_direction = 'Y'
    assert direction == expected_direction


def test_std_name_without_direction():
    standard_name_2 = 'atmosphere_optical_thickness_due_to_cloud'
    direction = parse_vector_axis(standard_name_2)
    assert direction is None


def test_std_name_with_transport_direction():
    standard_name_3 = 'ocean_heat_x_transport_due_to_diffusion'
    direction = parse_vector_axis(standard_name_3)
    expected_direction = 'X'
    assert direction == expected_direction
