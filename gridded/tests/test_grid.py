#!/usr/bin/env python

from __future__ import (absolute_import, division, print_function, unicode_literals)

import os
import pytest
import datetime as dt
import numpy as np
import datetime
import netCDF4 as nc
import pprint as pp
from gridded.grids import Grid, Grid_U, Grid_S


@pytest.fixture()
def sg_data():
    base_dir = os.path.dirname(__file__)
    s_data = os.path.join(base_dir, 'data')
    filename = os.path.join(s_data, 'staggered_sine_channel.nc')
    return filename, nc.Dataset(filename)

@pytest.fixture()
def sg_topology():
    return None

@pytest.fixture()
def sg():
    return Grid.from_netCDF(sg_data()[0], sg_data()[1], grid_topology=sg_topology())

@pytest.fixture()
def ug_data():
    base_dir = os.path.dirname(__file__)
    s_data = os.path.join(base_dir, 'data')
    filename = os.path.join(s_data, 'tri_ring.nc')
    return filename, nc.Dataset(filename)

@pytest.fixture()
def ug_topology():
    return None

@pytest.fixture()
def ug():
    return Grid.from_netCDF(ug_data()[0], ug_data()[1], grid_topology=ug_topology())

class TestPyGrid_S:
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
