"""
Created on Apr 7, 2015

@author: ayan

"""

from __future__ import (absolute_import, division, print_function)

import os
import tempfile

import pytest
import numpy as np
from netCDF4 import Dataset

from gridded.pysgrid.lookup import (LON_GRID_CELL_CENTER_LONG_NAME,
                                    LAT_GRID_CELL_CENTER_LONG_NAME,
                                    LON_GRID_CELL_NODE_LONG_NAME,
                                    LAT_GRID_CELL_NODE_LONG_NAME)


@pytest.fixture
def deltares_sgrid_no_optional_attr():
    fname = tempfile.mktemp(suffix='.nc')
    nc = Dataset(fname, 'w')
    # Define dimensions.
    nc.createDimension('MMAXZ', 4)
    nc.createDimension('NMAXZ', 4)
    nc.createDimension('MMAX', 4)
    nc.createDimension('NMAX', 4)
    nc.createDimension('KMAX', 2)
    nc.createDimension('KMAX1', 3)
    nc.createDimension('time', 2)
    # Define variables.
    xcor = nc.createVariable('XCOR', 'f4', ('MMAX', 'NMAX'))  # nodes
    ycor = nc.createVariable('YCOR', 'f4', ('MMAX', 'NMAX'))  # nodes
    xz = nc.createVariable('XZ', 'f4', ('MMAXZ', 'NMAXZ'))  # centers
    yz = nc.createVariable('YZ', 'f4', ('MMAXZ', 'NMAXZ'))  # centers
    u1 = nc.createVariable('U1', 'f4', ('time', 'KMAX', 'MMAX', 'NMAXZ'))
    v1 = nc.createVariable('V1', 'f4', ('time', 'KMAX', 'MMAXZ', 'NMAX'))
    w = nc.createVariable('W', 'f4', ('time', 'KMAX1', 'MMAXZ', 'NMAXZ'))
    times = nc.createVariable('time', 'f8', ('time',))
    grid = nc.createVariable('grid', 'i4')
    # Define variable attributes.
    grid.cf_role = 'grid_topology'
    grid.topology_dimension = 2
    grid.node_dimensions = 'MMAX NMAX'
    grid.face_dimensions = 'MMAXZ: MMAX (padding: low) NMAXZ: NMAX (padding: low)'  # noqa
    grid.vertical_dimensions = 'KMAX: KMAX1 (padding: none)'
    xcor.standard_name = 'projection_x_coordinate'
    xcor.long_name = 'X-coordinate of grid points'
    ycor.standard_name = 'projection_y_coordinate'
    ycor.long_name = 'Y-coordinate of grid points'
    xz.standard_name = 'projection_x_coordinate'
    xz.long_name = 'X-coordinate of cell centres'
    yz.standard_name = 'projection_y_coordinate'
    yz.long_name = 'Y-coordinate of cell centres'
    times.standard_name = 'time'
    u1.grid = 'some grid'
    u1.axes = 'X: NMAXZ Y: MMAX Z: KMAX'
    u1.standard_name = 'sea_water_x_velocity'
    v1.grid = 'some grid'
    v1.axes = 'X: NMAX Y: MMAXZ Z: KMAX'
    v1.standard_name = 'sea_water_y_velocity'
    w.grid = 'grid'
    w.location = 'face'
    # Create variable data.
    xcor[:] = np.random.random((4, 4))
    ycor[:] = np.random.random((4, 4))
    xz[:] = np.random.random((4, 4))
    yz[:] = np.random.random((4, 4))
    u1[:] = np.random.random((2, 2, 4, 4))
    v1[:] = np.random.random((2, 2, 4, 4))
    times[:] = np.random.random((2,))
    nc.sync()
    yield nc
    nc.close()
    os.remove(fname)


@pytest.fixture
def deltares_sgrid():
    """
    Create a netCDF file that is structurally similar to
    deltares output. Dimension and variable names may differ
    from an actual file.

    """
    fname = tempfile.mktemp(suffix='.nc')
    nc = Dataset(fname, 'w')
    # Define dimensions.
    nc.createDimension('MMAXZ', 4)
    nc.createDimension('NMAXZ', 4)
    nc.createDimension('MMAX', 4)
    nc.createDimension('NMAX', 4)
    nc.createDimension('KMAX', 2)
    nc.createDimension('KMAX1', 3)
    nc.createDimension('time', 2)
    # Define variables.
    xcor = nc.createVariable('XCOR', 'f4', ('MMAX', 'NMAX'))  # Nodes.
    ycor = nc.createVariable('YCOR', 'f4', ('MMAX', 'NMAX'))  # Nodes.
    xz = nc.createVariable('XZ', 'f4', ('MMAXZ', 'NMAXZ'))  # Centers.
    yz = nc.createVariable('YZ', 'f4', ('MMAXZ', 'NMAXZ'))  # Centers.
    u1 = nc.createVariable('U1', 'f4', ('time', 'KMAX', 'MMAX', 'NMAXZ'))
    fake_u1 = nc.createVariable('FAKE_U1', 'f4', ('time', 'KMAX', 'MMAX', 'NMAXZ'))  # noqa
    v1 = nc.createVariable('V1', 'f4', ('time', 'KMAX', 'MMAXZ', 'NMAX'))
    w = nc.createVariable('W', 'f4', ('time', 'KMAX1', 'MMAXZ', 'NMAXZ'))
    fake_w = nc.createVariable('FAKE_W', 'f4', ('time', 'MMAXZ', 'NMAXZ'))
    times = nc.createVariable('time', 'f8', ('time',))
    grid = nc.createVariable('grid', 'i4')
    latitude = nc.createVariable('latitude', 'f4', ('MMAXZ', 'NMAXZ'))
    longitude = nc.createVariable('longitude', 'f4', ('MMAXZ', 'NMAXZ'))
    grid_latitude = nc.createVariable('grid_latitude', 'f4', ('MMAX', 'NMAX'))  # noqa
    grid_longitude = nc.createVariable('grid_longitude', 'f4', ('MMAX', 'NMAX'))  # noqa
    # Define variable attributes.
    grid.cf_role = 'grid_topology'
    grid.topology_dimension = 2
    grid.node_dimensions = 'MMAX NMAX'
    grid.face_dimensions = 'MMAXZ: MMAX (padding: low) NMAXZ: NMAX (padding: low)'  # noqa
    grid.node_coordinates = 'XCOR YCOR'
    grid.face_coordinates = 'XZ YZ'
    grid.vertical_dimensions = 'KMAX: KMAX1 (padding: none)'
    latitude.long_name = LAT_GRID_CELL_CENTER_LONG_NAME[1]
    latitude.axes = 'X: NMAXZ Y: MMAXZ'
    longitude.long_name = LON_GRID_CELL_CENTER_LONG_NAME[1]
    longitude.axes = 'X: NMAXZ Y: MMAXZ'
    grid_latitude.long_name = LAT_GRID_CELL_NODE_LONG_NAME[1]
    grid_latitude.axes = 'X: NMAX Y: MMAX'
    grid_longitude.long_name = LON_GRID_CELL_NODE_LONG_NAME[1]
    grid_longitude.axes = 'X: NMAX Y: MMAX'
    times.standard_name = 'time'
    u1.grid = 'some grid'
    u1.axes = 'X: NMAXZ Y: MMAX Z: KMAX'
    u1.standard_name = 'sea_water_x_velocity'
    fake_u1.grid = 'some grid'
    v1.grid = 'some grid'
    v1.axes = 'X: NMAX Y: MMAXZ Z: KMAX'
    v1.standard_name = 'sea_water_y_velocity'
    w.grid = 'grid'
    w.location = 'face'
    fake_w.grid = 'grid'
    # Create variable data.
    xcor[:] = np.random.random((4, 4))
    ycor[:] = np.random.random((4, 4))
    xz[:] = np.random.random((4, 4))
    yz[:] = np.random.random((4, 4))
    u1[:] = np.random.random((2, 2, 4, 4))
    fake_u1[:] = np.random.random((2, 2, 4, 4))
    v1[:] = np.random.random((2, 2, 4, 4))
    times[:] = np.random.random((2,))
    latitude[:] = np.random.random((4, 4))
    longitude[:] = np.random.random((4, 4))
    grid_latitude[:] = np.random.random((4, 4))
    grid_longitude[:] = np.random.random((4, 4))
    w[:] = np.random.random((2, 3, 4, 4))
    fake_w[:] = np.random.random((2, 4, 4))
    nc.sync()
    yield nc
    nc.close()
    os.remove(fname)


@pytest.fixture
def roms_sgrid():
    """
    Create a netCDF file that is structurally similar to
    ROMS output. Dimension and variable names may differ
    from an actual file.

    """
    fname = tempfile.mktemp(suffix='.nc')
    nc = Dataset(fname, 'w')
    # Set dimensions.
    nc.createDimension('s_rho', 2)
    nc.createDimension('s_w', 3)
    nc.createDimension('time', 2)
    nc.createDimension('xi_rho', 4)
    nc.createDimension('eta_rho', 4)
    nc.createDimension('xi_psi', 3)
    nc.createDimension('eta_psi', 3)
    nc.createDimension('xi_u', 3)
    nc.createDimension('eta_u', 4)
    nc.createDimension('xi_v', 4)
    nc.createDimension('eta_v', 3)
    # Create coordinate variables.
    z_centers = nc.createVariable('s_rho', 'i4', ('s_rho',))
    nc.createVariable('s_w', 'i4', ('s_w',))
    times = nc.createVariable('time', 'f8', ('time',))
    nc.createVariable('xi_rho', 'f4', ('xi_rho',))
    nc.createVariable('eta_rho', 'f4', ('eta_rho',))
    nc.createVariable('xi_psi', 'f4', ('xi_psi',))
    nc.createVariable('eta_psi', 'f4', ('eta_psi',))
    x_us = nc.createVariable('xi_u', 'f4', ('xi_u',))
    y_us = nc.createVariable('eta_u', 'f4', ('eta_u',))
    x_vs = nc.createVariable('xi_v', 'f4', ('xi_v',))
    y_vs = nc.createVariable('eta_v', 'f4', ('eta_v',))
    # Create other variables.
    grid = nc.createVariable('grid', 'i2')
    u = nc.createVariable('u', 'f4', ('time', 's_rho', 'eta_u', 'xi_u'))
    v = nc.createVariable('v', 'f4', ('time', 's_rho', 'eta_v', 'xi_v'))
    fake_u = nc.createVariable('fake_u', 'f4', ('time', 's_rho', 'eta_u', 'xi_u'))  # noqa
    lon_centers = nc.createVariable('lon_rho', 'f4', ('eta_rho', 'xi_rho'))
    lat_centers = nc.createVariable('lat_rho', 'f4', ('eta_rho', 'xi_rho'))
    lon_nodes = nc.createVariable('lon_psi', 'f4', ('eta_psi', 'xi_psi'))
    lat_nodes = nc.createVariable('lat_psi', 'f4', ('eta_psi', 'xi_psi'))
    lat_u = nc.createVariable('lat_u', 'f4', ('eta_u', 'xi_u'))
    lon_u = nc.createVariable('lon_u', 'f4', ('eta_u', 'xi_u'))
    lat_v = nc.createVariable('lat_v', 'f4', ('eta_v', 'xi_v'))
    lon_v = nc.createVariable('lon_v', 'f4', ('eta_v', 'xi_v'))
    salt = nc.createVariable('salt', 'f4', ('time', 's_rho', 'eta_rho', 'xi_rho'))  # noqa
    zeta = nc.createVariable('zeta', 'f4', ('time', 'eta_rho', 'xi_rho'))
    # Create variable attributes.
    lon_centers.long_name = LON_GRID_CELL_CENTER_LONG_NAME[0]
    lon_centers.standard_name = 'longitude'
    lon_centers.axes = 'X: xi_rho Y: eta_rho'
    lat_centers.long_name = LAT_GRID_CELL_CENTER_LONG_NAME[0]
    lat_centers.standard_name = 'latitude'
    lat_centers.axes = 'X: xi_rho Y: eta_rho'
    lon_nodes.long_name = LON_GRID_CELL_NODE_LONG_NAME[0]
    lon_nodes.axes = 'X: xi_psi Y: eta_psi'
    lat_nodes.long_name = LAT_GRID_CELL_NODE_LONG_NAME[0]
    lat_nodes.axes = 'X: xi_psi Y: eta_psi'
    times.standard_name = 'time'
    grid.cf_role = 'grid_topology'
    grid.topology_dimension = 2
    grid.node_dimensions = 'xi_psi eta_psi'
    grid.face_dimensions = 'xi_rho: xi_psi (padding: both) eta_rho: eta_psi (padding: both)'  # noqa
    grid.edge1_dimensions = 'xi_u: xi_psi eta_u: eta_psi (padding: both)'
    grid.edge2_dimensions = 'xi_v: xi_psi (padding: both) eta_v: eta_psi'
    grid.node_coordinates = 'lon_psi lat_psi'
    grid.face_coordinates = 'lon_rho lat_rho'
    grid.edge1_coordinates = 'lon_u lat_u'
    grid.edge2_coordinates = 'lon_v lat_v'
    grid.vertical_dimensions = 's_rho: s_w (padding: none)'
    salt.grid = 'grid'
    zeta.location = 'face'
    zeta.coordinates = 'time lat_rho lon_rho'
    u.grid = 'some grid'
    u.axes = 'X: xi_u Y: eta_u'
    u.coordinates = 'time s_rho lat_u lon_u '
    u.location = 'edge1'
    u.standard_name = 'sea_water_x_velocity'
    v.grid = 'some grid'
    v.axes = 'X: xi_v Y: eta_v'
    v.location = 'edge2'
    v.standard_name = 'sea_water_y_velocity'
    fake_u.grid = 'some grid'
    # Create coordinate data.
    z_centers[:] = np.random.random(size=(2,))
    times[:] = np.random.random(size=(2,))
    lon_centers[:, :] = np.random.random(size=(4, 4))
    lat_centers[:, :] = np.random.random(size=(4, 4))
    lon_nodes[:] = np.random.random(size=(3, 3))
    lat_nodes[:] = np.random.random(size=(3, 3))
    x_us[:] = np.random.random(size=(3,))
    y_us[:] = np.random.random(size=(4,))
    x_vs[:] = np.random.random(size=(4,))
    y_vs[:] = np.random.random(size=(3,))
    u[:] = np.random.random(size=(2, 2, 4, 3))  # x-directed velocities
    v[:] = np.random.random(size=(2, 2, 3, 4))  # y-directed velocities
    fake_u[:] = np.random.random(size=(2, 2, 4, 3))
    lat_u[:] = np.random.random(size=(4, 3))
    lon_u[:] = np.random.random(size=(4, 3))
    lat_v[:] = np.random.random(size=(3, 4))
    lon_v[:] = np.random.random(size=(3, 4))
    salt[:] = np.random.random(size=(2, 2, 4, 4))
    nc.sync()
    yield nc
    nc.close()
    os.remove(fname)


@pytest.fixture
def wrf_sgrid():
    fname = tempfile.mktemp(suffix='.nc')
    nc = Dataset(fname, 'w')
    nc.createDimension('Time', 2)
    nc.createDimension('DateStrLen', 3)
    nc.createDimension('west_east', 4)
    nc.createDimension('south_north', 5)
    nc.createDimension('west_east_stag', 5)
    nc.createDimension('bottom_top', 3)
    nc.createDimension('south_north_stag', 6)
    nc.createDimension('bottom_top_stag', 4)
    times = nc.createVariable('Times', np.dtype(str), ('Time', 'DateStrLen'))  # noqa
    xtimes = nc.createVariable('XTIME', 'f8', ('Time', ))
    us = nc.createVariable('U', 'f4', ('Time', 'bottom_top', 'south_north', 'west_east_stag'))  # noqa
    us.grid = 'grid'
    us.location = 'edge1'
    fake_u = nc.createVariable('FAKE_U', 'f4', ('Time', 'bottom_top', 'south_north', 'west_east_stag'))  # noqa
    fake_u.grid = 'grid'
    vs = nc.createVariable('V', 'f4', ('Time', 'bottom_top', 'south_north_stag', 'west_east'))  # noqa
    vs.grid = 'grid'
    vs.location = 'edge2'
    ws = nc.createVariable('W', 'f4', ('Time', 'bottom_top_stag', 'south_north', 'west_east'))  # noqa
    ws.grid = 'grid'
    ws.location = 'face'
    temps = nc.createVariable('T', 'f4', ('Time', 'bottom_top', 'south_north', 'west_east'))  # noqa
    temps.grid = 'grid'
    temps.location = 'face'
    snow = nc.createVariable('SNOW', 'f4', ('Time', 'south_north', 'west_east'))  # noqa
    snow.grid = 'grid'
    xlats = nc.createVariable('XLAT', 'f4', ('south_north', 'west_east'))
    xlongs = nc.createVariable('XLONG', 'f4', ('south_north', 'west_east'))
    znus = nc.createVariable('ZNU', 'f4', ('Time', 'bottom_top'))
    znws = nc.createVariable('ZNW', 'f4', ('Time', 'bottom_top_stag'))
    xtimes.standard_name = 'time'
    grid = nc.createVariable('grid', 'i2')
    grid.cf_role = 'grid_topology'
    grid.topology_dimension = 2
    grid.node_dimensions = 'west_east_stag south_north_stag'
    grid.face_dimensions = ('west_east: west_east_stag (padding: none) '
                            'south_north: south_north_stag (padding: none)'
                            )
    grid.face_coordinates = 'XLONG XLAT'
    grid.vertical_dimensions = 'bottom_top: bottom_top_stag (padding: none)'  # noqa
    grid.edge1_dimensions = 'west_east_stag south_north: south_north_stag (padding: none)'  # noqa
    grid.edge2_dimensions = 'west_east: west_east_stag (padding: none) south_north_stag'  # noqa
    times[:] = np.random.random(size=(2, 3)).astype(str)
    xtimes[:] = np.random.random(size=(2,))
    us[:, :, :, :] = np.random.random(size=(2, 3, 5, 5))
    fake_u[:, :, :, :] = np.random.random(size=(2, 3, 5, 5))
    vs[:, :, :, :] = np.random.random(size=(2, 3, 6, 4))
    ws[:, :, :, :] = np.random.random(size=(2, 4, 5, 4))
    temps[:, :, :, :] = np.random.random(size=(2, 3, 5, 4))
    snow[:, :, :] = np.random.random(size=(2, 5, 4))
    xlats[:, :] = np.random.random(size=(5, 4))
    xlongs[:, :] = np.random.random(size=(5, 4))
    znus[:, :] = np.random.random(size=(2, 3))
    znws[:, :] = np.random.random(size=(2, 4))
    nc.sync()
    yield nc
    nc.close()
    os.remove(fname)


@pytest.fixture
def non_compliant_sgrid():
    """
    Create a netCDF file that is structurally similar to
    ROMS output. Dimension and variable names may differ
    from an actual file.

    """
    fname = tempfile.mktemp(suffix='.nc')
    nc = Dataset(fname, 'w')
    nc.createDimension('z_center', 2)
    nc.createDimension('z_node', 3)
    nc.createDimension('time', 2)
    nc.createDimension('x_center', 4)
    nc.createDimension('y_center', 4)
    nc.createDimension('x_node', 3)
    nc.createDimension('y_node', 3)
    nc.createDimension('x_u', 3)
    nc.createDimension('y_u', 4)
    nc.createDimension('x_v', 4)
    nc.createDimension('y_v', 3)
    # Create coordinate variables.
    z_centers = nc.createVariable('z_center', 'i4', ('z_center',))
    nc.createVariable('z_node', 'i4', ('z_node',))
    times = nc.createVariable('time', 'f8', ('time',))
    nc.createVariable('x_center', 'f4', ('x_center',))
    nc.createVariable('y_center', 'f4', ('y_center',))
    nc.createVariable('x_node', 'f4', ('x_node',))
    nc.createVariable('y_node', 'f4', ('y_node',))
    x_us = nc.createVariable('x_u', 'f4', ('x_u',))
    y_us = nc.createVariable('y_u', 'f4', ('y_u',))
    x_vs = nc.createVariable('x_v', 'f4', ('x_v',))
    y_vs = nc.createVariable('y_v', 'f4', ('y_v',))
    # Create other variables.
    grid = nc.createVariable('grid', 'i2')
    u = nc.createVariable('u', 'f4', ('time', 'z_center', 'y_u', 'x_u'))
    v = nc.createVariable('v', 'f4', ('time', 'z_center', 'y_v', 'x_v'))
    lon_centers = nc.createVariable('lon_center', 'f4', ('y_center', 'x_center'))  # noqa
    lat_centers = nc.createVariable('lat_center', 'f4', ('y_center', 'x_center'))  # noqa
    lon_nodes = nc.createVariable('lon_node', 'f4', ('y_node', 'x_node'))
    lat_nodes = nc.createVariable('lat_node', 'f4', ('y_node', 'x_node'))
    lat_u = nc.createVariable('lat_u', 'f4', ('y_u', 'x_u'))
    lon_u = nc.createVariable('lon_u', 'f4', ('y_u', 'x_u'))
    lat_v = nc.createVariable('lat_v', 'f4', ('y_v', 'x_v'))
    lon_v = nc.createVariable('lon_v', 'f4', ('y_v', 'x_v'))
    # Create variable attributes.
    lon_centers.long_name = LON_GRID_CELL_CENTER_LONG_NAME[0]
    lat_centers.long_name = LAT_GRID_CELL_CENTER_LONG_NAME[0]
    lon_nodes.long_name = LON_GRID_CELL_NODE_LONG_NAME[0]
    lat_nodes.long_name = LAT_GRID_CELL_NODE_LONG_NAME[0]
    grid.topology_dimension = 2
    grid.node_dimensions = 'x_node y_node'
    grid.face_dimensions = 'x_center: x_node (padding: both) y_center: y_node (padding: both)'  # noqa
    grid.edge1_dimensions = 'x_u: x_node y_u: y_node (padding: both)'
    grid.edge2_dimensions = 'x_v: x_node (padding: both) y_v: y_node'
    grid.node_coordinates = 'lon_node lat_node'
    grid.face_coordinates = 'lon_center lat_center'
    grid.edge1_coordinates = 'lon_u lat_u'
    grid.edge2_coordinates = 'lon_v lat_v'
    grid.vertical_dimensions = 'z_center: z_node (padding: none)'
    # Create coordinate data.
    z_centers[:] = np.random.random(size=(2,))
    times[:] = np.random.random(size=(2,))
    lon_centers[:, :] = np.random.random(size=(4, 4))
    lat_centers[:, :] = np.random.random(size=(4, 4))
    lon_nodes[:] = np.random.random(size=(3, 3))
    lat_nodes[:] = np.random.random(size=(3, 3))
    x_us[:] = np.random.random(size=(3,))
    y_us[:] = np.random.random(size=(4,))
    x_vs[:] = np.random.random(size=(4,))
    y_vs[:] = np.random.random(size=(3,))
    # X-directed velocities.
    u[:, :, :, :] = np.random.random(size=(2, 2, 4, 3))
    # Y-directed velocities.
    v[:] = np.random.random(size=(2, 2, 3, 4))
    lat_u[:] = np.random.random(size=(4, 3))
    lon_u[:] = np.random.random(size=(4, 3))
    lat_v[:] = np.random.random(size=(3, 4))
    lon_v[:] = np.random.random(size=(3, 4))
    nc.sync()
    yield nc
    nc.close()
    os.remove(fname)
