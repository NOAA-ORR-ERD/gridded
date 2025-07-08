"""
Created on Apr 7, 2015

@author: ayan

"""


import pytest
import numpy as np

from gridded.pysgrid.sgrid import SGrid, load_grid

from .write_nc_test_files import (deltares_sgrid,
                                  deltares_sgrid_no_optional_attr)

pytestmark = pytest.mark.skipif(True, reason="Lots of issues with these -- needs revisiting")


"""
Test SGrid No Coordinates.

Test to make sure that if no coordinates (e.g. face, edge1, etc)
are specified, those coordinates can be inferred from the dataset.

A file is representing a delft3d dataset is used for this test.

"""


@pytest.fixture
def sgrid(deltares_sgrid_no_optional_attr):
    return load_grid(deltares_sgrid_no_optional_attr)


def test_face_coordinate_inference(sgrid):
    face_coordinates = sgrid.face_coordinates
    expected_face_coordinates = (u'XZ', u'YZ')
    assert face_coordinates == expected_face_coordinates


def test_center_lon_deltares_no_coord(sgrid):
    center_lon = sgrid.center_lon
    assert center_lon.shape == (4, 4)


def test_center_lat_deltares_no_coord(sgrid):
    center_lat = sgrid.center_lat
    assert center_lat.shape == (4, 4)


def test_node_lon(sgrid):
    node_lon = sgrid.node_lon
    assert node_lon.shape == (4, 4)


def test_node_lat(sgrid):
    node_lat = sgrid.node_lat
    assert node_lat.shape == (4, 4)


def test_grid_angles(sgrid):
    angles = sgrid.angles
    angles_shape = (4, 4)
    assert angles.shape == angles_shape


"""
Test SGrid Delft3d Dataset.

"""


@pytest.fixture
def sgrid_obj(deltares_sgrid):
    return load_grid(deltares_sgrid)


def test_center_lon_deltares(sgrid_obj):
    center_lon = sgrid_obj.center_lon
    assert center_lon.shape == (4, 4)


def test_center_lat_deltares(sgrid_obj):
    center_lat = sgrid_obj.center_lat
    assert center_lat.shape == (4, 4)


def test_topology_dimension_deltares(sgrid_obj):
    topology_dim = sgrid_obj.topology_dimension
    expected_dim = 2
    assert topology_dim == expected_dim


def test_variable_slice(sgrid_obj):
    u_center_slices = sgrid_obj.U1.center_slicing
    v_center_slices = sgrid_obj.V1.center_slicing
    u_center_expected = (np.s_[:], np.s_[:], np.s_[:], np.s_[1:])
    v_center_expected = (np.s_[:], np.s_[:], np.s_[1:], np.s_[:])
    xz_center_slices = sgrid_obj.XZ.center_slicing
    xcor_center_slices = sgrid_obj.XCOR.center_slicing
    xz_center_expected = (np.s_[1:], np.s_[1:])
    xcor_center_expected = (np.s_[:], np.s_[:])
    assert u_center_slices == u_center_expected
    assert v_center_slices == v_center_expected
    assert xz_center_slices == xz_center_expected
    assert xcor_center_slices == xcor_center_expected


def test_averaging_axes(sgrid_obj):
    u1c_axis = sgrid_obj.U1.center_axis
    u1c_expected = 0
    v1n_axis = sgrid_obj.V1.node_axis
    v1n_expected = 0
    latitude_c_axis = sgrid_obj.latitude.center_axis
    latitude_n_axis = sgrid_obj.latitude.node_axis
    assert u1c_axis == u1c_expected
    assert v1n_axis == v1n_expected
    assert latitude_c_axis is None
    assert latitude_n_axis is None


def test_grid_optional_attrs(sgrid_obj):
    face_coordinates = sgrid_obj.face_coordinates
    node_coordinates = sgrid_obj.node_coordinates
    edge1_coordinates = sgrid_obj.edge1_coordinates
    edge2_coordinates = sgrid_obj.edge2_coordinates
    fc_expected = ('XZ', 'YZ')
    nc_expected = ('XCOR', 'YCOR')
    assert face_coordinates == fc_expected
    assert node_coordinates == nc_expected
    assert edge1_coordinates is None
    assert edge2_coordinates is None


def test_grid_variables_deltares(sgrid_obj):
    grid_variables = sgrid_obj.grid_variables
    expected_grid_variables = [u'U1', u'V1', u'FAKE_U1', u'W', u'FAKE_W']
    assert set(grid_variables) == set(expected_grid_variables)


def test_angles(sgrid_obj):
    angles = sgrid_obj.angles
    expected_shape = (4, 4)
    assert angles.shape == expected_shape


def test_no_3d_attributes(sgrid_obj):
    assert not hasattr(sgrid_obj, 'volume_padding')
    assert not hasattr(sgrid_obj, 'volume_dimensions')
    assert not hasattr(sgrid_obj, 'volume_coordinates')
    assert not hasattr(sgrid_obj, 'face1_padding')
    assert not hasattr(sgrid_obj, 'face1_coordinates')
    assert not hasattr(sgrid_obj, 'face1_dimensions')
    assert not hasattr(sgrid_obj, 'face2_padding')
    assert not hasattr(sgrid_obj, 'face2_coordinates')
    assert not hasattr(sgrid_obj, 'face2_dimensions')
    assert not hasattr(sgrid_obj, 'face3_padding')
    assert not hasattr(sgrid_obj, 'face3_coordinates')
    assert not hasattr(sgrid_obj, 'edge3_padding')
    assert not hasattr(sgrid_obj, 'edge3_coordinates')
    assert not hasattr(sgrid_obj, 'edge3_dimensions')


def test_2d_attributes(sgrid_obj):
    assert hasattr(sgrid_obj, 'face_padding')
    assert hasattr(sgrid_obj, 'face_coordinates')
    assert hasattr(sgrid_obj, 'face_dimensions')
    assert hasattr(sgrid_obj, 'vertical_padding')
    assert hasattr(sgrid_obj, 'vertical_dimensions')


"""
Test SGrid Save Node Coordinates.

Test that `SGrid.save_as_netcdf `is saving the content correctly.

"""


def test_round_trip(deltares_sgrid, tmpdir):
    """
    TODO: add more "round-trip" tests.

    """
    fname = tmpdir.mkdir('files').join('deltares_roundtrip.nc')
    sg_obj = load_grid(deltares_sgrid)
    sg_obj.save_as_netcdf(fname)


@pytest.fixture
def sgrid_no_node(deltares_sgrid):
    return load_grid(deltares_sgrid)


def test_save_as_netcdf(sgrid_no_node):
    """
    Test that the attributes in the
    saved netCDF file are as expected.

    """
    target_dims = sgrid_no_node.dimensions
    expected_target_dims = [(u'MMAXZ', 4),
                            (u'NMAXZ', 4),
                            (u'MMAX', 4),
                            (u'NMAX', 4),
                            (u'KMAX', 2),
                            (u'KMAX1', 3),
                            (u'time', 2)
                            ]
    target_vars = sgrid_no_node.variables
    expected_target_vars = [u'XZ',
                            u'YZ',
                            u'XCOR',
                            u'YCOR',
                            u'grid',
                            u'U1',
                            u'FAKE_U1',
                            u'V1',
                            u'W',
                            u'FAKE_W',
                            u'time',
                            u'latitude',
                            u'longitude',
                            u'grid_latitude',
                            u'grid_longitude'
                            ]
    target_grid_vars = sgrid_no_node.grid_variables
    expected_target_grid_vars = [u'U1',
                                 u'FAKE_U1',
                                 u'V1',
                                 u'W',
                                 u'FAKE_W']
    target_face_coordinates = sgrid_no_node.face_coordinates
    expected_target_face_coordinates = (u'XZ', u'YZ')
    assert isinstance(sgrid_no_node, SGrid)
    assert len(target_dims) == len(expected_target_dims)
    assert set(target_dims) == set(expected_target_dims)
    assert len(target_vars) == len(expected_target_vars)
    assert set(target_vars) == set(expected_target_vars)
    assert len(target_grid_vars) == len(expected_target_grid_vars)
    assert set(target_grid_vars) == set(expected_target_grid_vars)
    assert target_face_coordinates == expected_target_face_coordinates


def test_saved_sgrid_attributes(sgrid_no_node):
    """
    Test that calculated/inferred attributes
    are as expected from the saved filed.

    """
    u1_var = sgrid_no_node.U1
    u1_var_center_avg_axis = u1_var.center_axis
    expected_u1_center_axis = 0
    u1_vector_axis = u1_var.vector_axis
    expected_u1_vector_axis = 'X'
    original_angles = sgrid_no_node.angles
    saved_angles = sgrid_no_node.angles
    assert u1_var_center_avg_axis == expected_u1_center_axis
    assert u1_vector_axis == expected_u1_vector_axis
    np.testing.assert_almost_equal(original_angles, saved_angles, decimal=3)  # noqa
