#!/usr/bin/env python

from __future__ import (absolute_import, division, print_function, unicode_literals)

import os
import pytest
import numpy as np
import netCDF4 as nc

from gridded.grids import Grid, Grid_U, Grid_S, Grid_R
from .utilities import get_test_file_dir


@pytest.fixture()
def sg_data():
    filename = os.path.join(get_test_file_dir(), 'staggered_sine_channel.nc')
    return filename, nc.Dataset(filename)


@pytest.fixture()
def sg_topology():
    return None


@pytest.fixture()
def sg():
    return Grid.from_netCDF(sg_data()[0], sg_data()[1], grid_topology=sg_topology())


@pytest.fixture()
def ug_data():
    filename = os.path.join(get_test_file_dir(), 'tri_ring.nc')
    return filename, nc.Dataset(filename)


@pytest.fixture()
def ug_topology():
    return None


@pytest.fixture()
def ug():
    return Grid.from_netCDF(ug_data()[0], ug_data()[1], grid_topology=ug_topology())


class TestGrid_S:
    def test_construction(self, sg_data, sg_topology):
        filename = sg_data[0]
        dataset = sg_data[1]
        grid_topology = sg_topology
        sg = Grid_S.from_netCDF(filename, dataset, grid_topology=grid_topology)
        assert sg.filename == filename

        sg2 = Grid_S.from_netCDF(filename)
        assert sg2.filename == filename

        sg3 = Grid.from_netCDF(filename, dataset, grid_topology=grid_topology)
        sg4 = Grid.from_netCDF(filename)
        print(sg3.shape)
        print(sg4.shape)
        assert sg == sg3
        assert sg2 == sg4

    def test_masked_grid(self, sg_data, sg_topology):
        filename = sg_data[0]
        dataset = sg_data[1]
        grid_topology = sg_topology
        sg = Grid_S.from_netCDF(filename, dataset, grid_topology=grid_topology)

        assert sg.node_mask is not None
        assert all(sg.node_mask[0,:] == True)
        assert all(sg.edge1_mask[-1,:] == True)

        sg.build_celltree(grid='node', use_mask=False)
        assert sg._cell_trees['node'][1].shape == sg.nodes.reshape(-1,2).shape
        on_grid_result = sg.locate_faces([0.5,0], 'node')
        off_grid_result = sg.locate_faces([0,0.5], 'node')
        masked_territory_result = sg.locate_faces([0.5,0.9], 'node')
        sg.build_celltree(grid='node', use_mask=True)
        assert sg._cell_trees['node'][1].shape == (48,2)
        #locate a point that is on the grid in unmasked territory, and make
        #sure that with or without the mask returns the same result
        assert all(sg.locate_faces([0.5,0], 'node') == on_grid_result)
        #locate a point off-grid, and make sure the same result is returned
        assert all(sg.locate_faces([0,0.5], 'node') == off_grid_result)
        #masked territory should be different.
        assert all(sg.locate_faces([0.5,0.9], 'node') != masked_territory_result)

        #rebuild without the mask, and make sure results match
        sg.build_celltree(grid='node', use_mask=False)
        assert all(sg.locate_faces([0.5,0], 'node') == on_grid_result)
        assert all(sg.locate_faces([0,0.5], 'node') == off_grid_result)
        assert all(sg.locate_faces([0.5,0.9], 'node') == masked_territory_result)

        sg.build_celltree(grid='node', use_mask=True)
        sg.use_masked_boundary=True
        sg.build_celltree(grid='node', use_mask=True)
        #because masked nodes that are adjacent to at least one unmasked node
        #now get unmasked, and this grid has a one-node-thick border,
        #all nodes should be unmasked
        assert len(np.where(sg._masks['node'][0])[0]) == 0
        #behavior should be identical as well
        assert all(sg.locate_faces([0.5,0], 'node') == on_grid_result)
        assert all(sg.locate_faces([0,0.5], 'node') == off_grid_result)
        assert all(sg.locate_faces([0.5,0.9], 'node') == masked_territory_result)

        #rerun with center just to make sure
        sg.use_masked_boundary=False
        sg.build_celltree(grid='center', use_mask=False)
        assert sg._cell_trees['center'][1].shape == sg.centers.reshape(-1,2).shape
        on_grid_result = sg.locate_faces([0.3,0], 'center')
        off_grid_result = sg.locate_faces([-1,0.5], 'center')
        masked_territory_result = sg.locate_faces([0.1,-0.9], 'center')
        sg.build_celltree(grid='center', use_mask=True)
        #locate a point that is on the grid in unmasked territory, and make
        #sure that with or without the mask returns the same result
        assert all(sg.locate_faces([0.5,0], 'center') == on_grid_result)
        #locate a point off-grid, and make sure the same result is returned
        assert all(sg.locate_faces([-1,0.5], 'center') == off_grid_result)
        #masked territory should be different.
        assert all(sg.locate_faces([0.1, -0.9], 'center') != masked_territory_result)



class TestPyGrid_U:
    def test_construction(self, ug_data, ug_topology):
        filename = ug_data[0]
        dataset = ug_data[1]
        grid_topology = ug_topology
        ug = Grid_U.from_netCDF(filename, dataset, grid_topology=grid_topology)
#         assert ug.filename == filename
#         assert isinstance(ug.node_lon, nc.Variable)
#         assert ug.node_lon.name == 'lonc'

        ug2 = Grid_U.from_netCDF(filename)
        assert ug2.filename == filename
#         assert isinstance(ug2.node_lon, nc.Variable)
#         assert ug2.node_lon.name == 'lon'

        ug3 = Grid.from_netCDF(filename, dataset, grid_topology=grid_topology)
        ug4 = Grid.from_netCDF(filename)
        print(ug3.shape)
        print(ug4.shape)
        assert ug == ug3
        assert ug2 == ug4


@pytest.fixture()
def rg_data():
    lons = np.array((0,10,20,30,40,55))
    lats = np.array((0,2,3,4,5,7,9))
    return lons, lats


@pytest.fixture()
def example_rg():
    lons = np.array((0,10,20,30,40,55))
    lats = np.array((0,2,3,4,5,7,9))
    rg = Grid_R(node_lon=lons,
                node_lat=lats)
    return rg


class TestGrid_R:
    def test_construction(self, rg_data):
        node_lon = rg_data[0]
        node_lat = rg_data[1]
        rg = Grid_R(node_lon=node_lon,
                    node_lat=node_lat)

    def test_locate_faces(self, example_rg):
        points = np.array(([5,1],[6,1],[7,1],[-1,0],[42,0]))
        idxs = example_rg.locate_faces(points)
        answer = np.array(([0,0],[0,0],[0,0],[-1,-1],[4,0]))
        assert np.all(idxs == answer)

        points = np.array([5,1])
        idxs = example_rg.locate_faces(points)
        answer = np.array([0,0])
        assert np.all(idxs == answer)

    def test_interpolation(self, example_rg):
        example_rg.node_lon = np.array([0,1,2,5])
        example_rg.node_lat = np.array([0,1,2,12])
        points = np.array(([0.5,0.5],[3.5,2],[-1,0],[0,-1]))
        v1 = np.mgrid[0:4,0:4][1]
        val = example_rg.interpolate_var_to_points(points, v1, method='linear')
        assert np.all(np.isclose(val,np.array([0.5,2,0,0])))

        points = np.array([3.5,2])
        val = example_rg.interpolate_var_to_points(points, v1, method='linear')
        assert np.all(np.isclose(val,np.array([2])))
