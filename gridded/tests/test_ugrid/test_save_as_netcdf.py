#!/usr/bin/env python

"""
Tests for saving a UGrid in netcdf format.

Designed to be run with pytest.

"""

from __future__ import (absolute_import, division, print_function)

import os
import netCDF4
import numpy as np

import pytest

from gridded.grids import Grid_U
from gridded.variable import Variable
from gridded import Dataset

from .utilities import chdir, two_triangles, twenty_one_triangles

temp_files = os.path.join(os.path.dirname(__file__), 'temp_files')

# Set this to True if you want to see what gets written to debug
KEEP_TEMP_FILES = False
# KEEP_TEMP_FILES = True


# kludge to keep a counter going
file_counter = [0]
@pytest.fixture
def ncds():
    """
    provides a new netCDF4 Dataset
    JAH: This needs to be explained!
    """
    if not os.path.isdir(temp_files):
        os.mkdir(temp_files)
    fname = os.path.join(temp_files, 'temp_{}.nc'.format(file_counter[0]))
    file_counter[0] = file_counter[0] + 1    # remove file if it's already there
    # "clobber=True" should do this, but
    # it doesn't always clean up properly
    if os.path.isfile(fname):
        os.remove(fname)
    ncds = netCDF4.Dataset(fname,
                           mode="w",
                           clobber=True,
                           format="NETCDF4")
    yield (fname, ncds)

    if not ncds.isopen:
        ncds.close()

    if not KEEP_TEMP_FILES:
        try:
            os.remove(fname)
        except OSError:
            pass


def nc_has_variable(ds, var_name):
    """
    Checks that a netcdf file has the given variable defined.

    :param ds: a netCDF4.Dataset object, or a netcdf file name.

    """
    if not isinstance(ds, netCDF4.Dataset):
        ds = netCDF4.Dataset(ds)

    if var_name in ds.variables:
        return True
    else:
        print('{} is not a variable in the Dataset'.format(var_name))
        return False


def nc_has_dimension(ds, dim_name):
    """
    Checks that a netcdf file has the given dimension defined.

    :param ds: a netCDF4.Dataset object, or a netcdf file name.

    """
    if not isinstance(ds, netCDF4.Dataset):
        ds = netCDF4.Dataset(ds)
    if dim_name in ds.dimensions:
        return True
    else:
        return False


def nc_var_has_attr(ds, var_name, att_name):
    """
    Checks that the variable, var_name, has the attribute, att_name.

    """
    if not isinstance(ds, netCDF4.Dataset):
        ds = netCDF4.Dataset(ds)

    try:
        getattr(ds.variables[var_name], att_name)
        return True
    except AttributeError:
        return False


def nc_var_has_attr_vals(ds, var_name, att_dict):
    """
    Checks that the variable, var_name, as the attributes (and values)
    in the att_dict.

    """
    if not isinstance(ds, netCDF4.Dataset):
        ds = netCDF4.Dataset(ds)

    for key, val in att_dict.items():
        try:
            if val != getattr(ds.variables[var_name], key):
                return False
        except AttributeError:
            return False
    return True


def test_simple_write(two_triangles, ncds):
    grid = two_triangles

    fname, ncds = ncds

    grid.save_as_netcdf(ncds)
    ds = netCDF4.Dataset(fname)

    # TODO: Could be lots of tests here.
    assert nc_has_variable(ds, 'mesh')
    assert nc_var_has_attr_vals(ds, 'mesh', {
        'cf_role': 'mesh_topology',
        'topology_dimension': 2,
        'long_name': u'Topology data of 2D unstructured mesh'})
    ds.close()


def test_set_mesh_name(two_triangles, ncds):
    grid = two_triangles
    grid.mesh_name = 'mesh_2'

    fname, ncds = ncds

    grid.save_as_netcdf(ncds)
    ds = netCDF4.Dataset(fname)

    assert nc_has_variable(ds, 'mesh_2')
    assert nc_var_has_attr_vals(ds, 'mesh_2', {
        'cf_role': 'mesh_topology',
        'topology_dimension': 2,
        'long_name': u'Topology data of 2D unstructured mesh'})
    assert nc_var_has_attr_vals(ds, 'mesh_2', {
        'cf_role': 'mesh_topology',
        'topology_dimension': 2,
        'long_name': u'Topology data of 2D unstructured mesh',
        'node_coordinates': 'mesh_2_node_lon mesh_2_node_lat'})

    assert nc_has_variable(ds, 'mesh_2_node_lon')
    assert nc_has_variable(ds, 'mesh_2_node_lat')
    assert nc_has_variable(ds, 'mesh_2_face_nodes')
    assert nc_has_variable(ds, 'mesh_2_edge_nodes')

    assert nc_has_dimension(ds, 'mesh_2_num_node')
    assert nc_has_dimension(ds, 'mesh_2_num_edge')
    assert nc_has_dimension(ds, 'mesh_2_num_face')
    assert nc_has_dimension(ds, 'mesh_2_num_vertices')

    assert not nc_var_has_attr(ds, 'mesh_2', 'face_edge_connectivity')
    ds.close()


def test_write_with_depths(two_triangles, ncds):
    """Tests writing a netcdf file with depth data."""

    fname, ncds = ncds

    grid = two_triangles
    grid.mesh_name = 'mesh1'

    gds = Dataset(grid=grid)

    # Create a Variable object for the depths:
    depths = Variable(name='depth',
                      location='node',
                      data=[1.0, 2.0, 3.0, 4.0])
    depths.attributes['units'] = 'm'
    depths.attributes['standard_name'] = 'sea_floor_depth_below_geoid'
    depths.attributes['positive'] = 'down'

    gds.variables['depth'] = depths

    gds.save(ncds, format='netcdf4')

    ds = netCDF4.Dataset(fname)

    assert nc_has_variable(ds, 'mesh1')
    assert nc_has_variable(ds, 'depth')
    assert nc_var_has_attr_vals(ds, 'depth', {
        'coordinates': 'mesh1_node_lon mesh1_node_lat',
        'location': 'node',
        'mesh': 'mesh1'})
    ds.close()


def test_write_with_velocities(two_triangles, ncds):
    """Tests writing a netcdf file with velocities on the faces."""

    fname, ncds = ncds

    two_triangles.mesh_name = 'mesh2'

    gds = Dataset(grid=two_triangles)

    # Create a Variable object for u velocity:
    u_vel = Variable('u', location='face', data=[1.0, 2.0])
    u_vel.attributes['units'] = 'm/s'
    u_vel.attributes['standard_name'] = 'eastward_sea_water_velocity'

    gds.variables['u'] = u_vel

    # Create a Variable object for v velocity:
    v_vel = Variable('v', location='face', data=[3.2, 4.3])
    v_vel.attributes['units'] = 'm/s'
    v_vel.attributes['standard_name'] = 'northward_sea_water_velocity'

    gds.variables['v'] = v_vel

    # Add coordinates for face data.
    gds.grid.build_face_coordinates()


    gds.save(ncds)

    ds = netCDF4.Dataset(fname)

    assert nc_has_variable(ds, 'mesh2')
    assert nc_has_variable(ds, 'u')
    assert nc_has_variable(ds, 'v')
    assert nc_var_has_attr_vals(ds, 'u',
                                {'coordinates': 'mesh2_face_lon mesh2_face_lat',
                                 'location': 'face',
                                 'mesh': 'mesh2',
                                 }
                                )
    ds.close()


def test_write_with_edge_data(two_triangles, ncds):
    """Tests writing a netcdf file with data on the edges (fluxes, maybe?)."""

    fname, ncds = ncds

    two_triangles.mesh_name = 'mesh2'

    gds = Dataset(grid=two_triangles)


    # Create a Variable object for fluxes:
    flux = Variable('flux', location='edge', data=[0.0, 0.0, 4.1, 0.0, 5.1, ])
    flux.attributes['units'] = 'm^3/s'
    flux.attributes['long_name'] = 'volume flux between cells'
    flux.attributes['standard_name'] = 'ocean_volume_transport_across_line'

    gds.variables['flux'] = flux

    # Add coordinates for edges.
    gds.grid.build_edge_coordinates()

    gds.save(ncds)

    ds = netCDF4.Dataset(fname)

    assert nc_has_variable(ds, 'mesh2')
    assert nc_has_variable(ds, 'flux')
    assert nc_var_has_attr_vals(ds, 'flux', {
        'coordinates': 'mesh2_edge_lon mesh2_edge_lat',
        'location': 'edge',
        'units': 'm^3/s',
        'mesh': 'mesh2'})
    assert np.array_equal(ds.variables['mesh2_edge_lon'],
                          gds.grid.edge_coordinates[:, 0])
    assert np.array_equal(ds.variables['mesh2_edge_lat'],
                          gds.grid.edge_coordinates[:, 1])
    ds.close()

def test_write_with_bound_data(two_triangles, ncds):
    """
    Tests writing a netcdf file with data on the boundaries
    suitable for boundary conditions, for example fluxes.

    """
    fname, ncds = ncds
    grid = two_triangles
    grid.mesh_name = 'mesh3'

    gds = Dataset(grid=grid)

    # Add the boundary definitions:
    grid.boundaries = [(0, 1),
                       (0, 2),
                       (1, 3),
                       (2, 3)]

    # Create a Variable object for boundary conditions:
    bnds = Variable('bnd_cond', location='boundary', data=[0, 1, 0, 0])
    bnds.attributes['long_name'] = 'model boundary conditions'
    bnds.attributes['flag_values'] = '0 1'
    bnds.attributes['flag_meanings'] = 'no_flow_boundary  open_boundary'

    gds.variables['bnd_cond'] = bnds

    gds.save(ncds)

    ds = netCDF4.Dataset(fname)

    assert nc_has_variable(ds, 'mesh3')
    assert nc_has_variable(ds, 'bnd_cond')
    assert nc_var_has_attr_vals(ds, 'mesh3', {
        'boundary_node_connectivity': 'mesh3_boundary_nodes'})
    assert nc_var_has_attr_vals(ds,
                                'bnd_cond',
                                {'location': 'boundary',
                                 'flag_values': '0 1',
                                 'flag_meanings': 'no_flow_boundary  open_boundary',
                                 'mesh': 'mesh3',
                                 })
    # There should be no coordinates attribute or variable for the
    # boundaries as there is no boundaries_coordinates defined.
    assert not nc_has_variable(ds, 'mesh_boundary_lon')
    assert not nc_has_variable(ds, 'mesh_boundary_lat')
    assert not nc_var_has_attr(ds, 'bnd_cond', 'coordinates')
    ds.close()


def test_write_everything(twenty_one_triangles, ncds):
    """An example with all features enabled, and a less trivial grid."""

    fname, ncds = ncds
    grid = twenty_one_triangles
    grid.mesh_name = 'mesh'

    gds = Dataset(grid=grid)

    grid.build_edges()
    grid.build_face_face_connectivity()

    grid.build_edge_coordinates()
    grid.build_face_coordinates()
    grid.build_boundary_coordinates()

    # Depth on the nodes.
    depths = Variable('depth', location='node', data=np.linspace(1, 10, 20))
    depths.attributes['units'] = 'm'
    depths.attributes['standard_name'] = 'sea_floor_depth_below_geoid'
    depths.attributes['positive'] = 'down'
    gds.variables['depth'] = depths

    # Create a Variable object for u velocity:
    u_vel = Variable('u', location='face', data=np.sin(np.linspace(3, 12, 21)))
    u_vel.attributes['units'] = 'm/s'
    u_vel.attributes['standard_name'] = 'eastward_sea_water_velocity'

    gds.variables['u'] = u_vel

    # Create a Variable object for v velocity:
    v_vel = Variable('v', location='face', data=np.sin(np.linspace(12, 15, 21)))
    v_vel.attributes['units'] = 'm/s'
    v_vel.attributes['standard_name'] = 'northward_sea_water_velocity'

    gds.variables['v'] = v_vel

    # Fluxes on the edges:
    flux = Variable('flux', location='edge', data=np.linspace(1000, 2000, 41))
    flux.attributes['units'] = 'm^3/s'
    flux.attributes['long_name'] = 'volume flux between cells'
    flux.attributes['standard_name'] = 'ocean_volume_transport_across_line'

    gds.variables['flux'] = flux

    # Boundary conditions:
    bounds = np.zeros((19,), dtype=np.uint8)
    bounds[7] = 1
    bnds = Variable('bnd_cond', location='boundary', data=bounds)
    bnds.attributes['long_name'] = 'model boundary conditions'
    bnds.attributes['flag_values'] = '0 1'
    bnds.attributes['flag_meanings'] = 'no_flow_boundary  open_boundary'

    gds.variables['bnd_cond'] = bnds

    gds.save(ncds)

    ds = netCDF4.Dataset(fname)

    # Now the tests:
    assert nc_has_variable(ds, 'mesh')
    assert nc_has_variable(ds, 'depth')
    assert nc_var_has_attr_vals(ds, 'depth', {
        'coordinates': 'mesh_node_lon mesh_node_lat',
        'location': 'node'})
    assert nc_has_variable(ds, 'u')
    assert nc_has_variable(ds, 'v')
    assert nc_var_has_attr_vals(ds, 'u', {
        'coordinates': 'mesh_face_lon mesh_face_lat',
        'location': 'face',
        'mesh': 'mesh'})
    assert nc_var_has_attr_vals(ds, 'v', {
        'coordinates': 'mesh_face_lon mesh_face_lat',
        'location': 'face',
        'mesh': 'mesh'})
    assert nc_has_variable(ds, 'flux')
    assert nc_var_has_attr_vals(ds, 'flux', {
        'coordinates': 'mesh_edge_lon mesh_edge_lat',
        'location': 'edge',
        'units': 'm^3/s',
        'mesh': 'mesh'})
    assert nc_has_variable(ds, 'mesh')
    assert nc_has_variable(ds, 'bnd_cond')
    assert nc_var_has_attr_vals(ds, 'mesh', {
        'boundary_node_connectivity': 'mesh_boundary_nodes'})
    assert nc_var_has_attr_vals(ds, 'bnd_cond', {
        'location': 'boundary',
        'flag_values': '0 1',
        'flag_meanings': 'no_flow_boundary  open_boundary',
        'mesh': 'mesh'})
    ds.close()

    # And make sure pyugrid can reload it!
    gds = Dataset(fname)
    grid = gds.grid
    # And that some things are the same.
    # NOTE: more testing might be good here.
    # maybe some grid comparison functions?

    assert grid.mesh_name == 'mesh'
    assert len(grid.nodes) == 20

    depth = gds.variables['depth']
    assert depth.attributes['units'] == 'm'

    u = gds.variables['u']
    assert u.attributes['units'] == 'm/s'


if __name__ == "__main__":
    test_simple_write()
    test_set_mesh_name()
    test_write_with_depths()
    test_write_with_velocities()
    test_write_with_edge_data()
