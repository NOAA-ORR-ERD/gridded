#!/usr/bin/env python

import os
import pytest
import datetime as dt
import numpy as np
import datetime
import netCDF4 as nc
from ..gridded import PyGrid, PyGrid_U, PyGrid_S
import pprint as pp
from gridded import Grid


def test_init():
    """ tests you can intitize a basic datset"""
    G = Grid.from_netCDF(os.path.join('test_data', 'staggered_sine_channel.nc'))
    print G.node_lon

if __name__ == '__main__':
    test_init()
    print 'success'


@pytest.fixture()
def sg_data():
    base_dir = os.path.dirname(__file__)
    s_data = os.path.join(base_dir, 'test_data')
    filename = os.path.join(s_data, 'staggered_sine_channel.nc')
    return filename, nc.Dataset(filename)

@pytest.fixture()
def sg_topology():
    return None

@pytest.fixture()
def sg():
    return PyGrid.from_netCDF(sg_data()[0], sg_data()[1], grid_topology=sg_topology())

@pytest.fixture()
def ug_data():
    base_dir = os.path.dirname(__file__)
    s_data = os.path.join(base_dir, 'test_data')
    filename = os.path.join(s_data, 'tri_ring.nc')
    return filename, nc.Dataset(filename)

@pytest.fixture()
def ug_topology():
    pass

@pytest.fixture()
def ug():
    return PyGrid.from_netCDF(ug_data()[0], ug_data()[1], grid_topology=ug_topology())

class TestPyGrid_S:
    def test_construction(self, sg_data, sg_topology):
        filename = sg_data[0]
        dataset = sg_data[1]
        grid_topology = sg_topology
        sg = PyGrid_S.from_netCDF(filename, dataset, grid_topology=grid_topology)
        assert sg.filename == filename

        sg2 = PyGrid_S.from_netCDF(filename)
        assert sg2.filename == filename

        sg3 = PyGrid.from_netCDF(filename, dataset, grid_topology=grid_topology)
        sg4 = PyGrid.from_netCDF(filename)
        print sg3.shape
        print sg4.shape
        assert sg == sg3
        assert sg2 == sg4


class TestPyGrid_U:
    def test_construction(self, ug_data, ug_topology):
        filename = ug_data[0]
        dataset = ug_data[1]
        grid_topology = ug_topology
        ug = PyGrid_U.from_netCDF(filename, dataset, grid_topology=grid_topology)
#         assert ug.filename == filename
#         assert isinstance(ug.node_lon, nc.Variable)
#         assert ug.node_lon.name == 'lonc'

        ug2 = PyGrid_U.from_netCDF(filename)
        assert ug2.filename == filename
#         assert isinstance(ug2.node_lon, nc.Variable)
#         assert ug2.node_lon.name == 'lon'

        ug3 = PyGrid.from_netCDF(filename, dataset, grid_topology=grid_topology)
        ug4 = PyGrid.from_netCDF(filename)
        print ug3.shape
        print ug4.shape
        assert ug == ug3
        assert ug2 == ug4
