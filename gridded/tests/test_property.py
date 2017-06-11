#!/usr/bin/env python

# py2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import sys
import datetime

import pytest
import numpy as np
import netCDF4 as nc

from gridded.variable import Variable, VectorVariable
from gridded.tests.utilities import get_test_file_dir

from gridded.grids import Grid

test_dir = get_test_file_dir()


'''
Need to hook this up to existing test data infrastructure
'''
@pytest.fixture()
def sg_data():
    s_data = get_test_file_dir()
    print(s_data)
    filename = os.path.join(s_data, 'staggered_sine_channel.nc')
    print(filename)
    return filename, nc.Dataset(filename)


class TestVariable:


    def test_construction(self, sg_data):

        fn = sg_data[0]
        sinusoid = sg_data[1]
        data = sinusoid['u'][:]
        grid = Grid.from_netCDF(dataset=sinusoid)
        time = None

        u = Variable(name='u',
                        units='m/s',
                        data=data,
                        grid=grid,
                        time=time,
                        data_file='staggered_sine_channel.nc',
                        grid_file='staggered_sine_channel.nc')

        curr_file = os.path.join(test_dir, 'staggered_sine_channel.nc')
        k = Variable.from_netCDF(filename=curr_file, varname='u', name='u')
        assert k.name == u.name
        assert k.units == 'm/s'
        # fixme: this was failing
        # assert k.time == u.time
        assert k.data[0, 0] == u.data[0, 0]

    def test_at(self):
        curr_file = os.path.join(test_dir, 'staggered_sine_channel.nc')
        u = Variable.from_netCDF(filename=curr_file, varname='u_rho')
        v = Variable.from_netCDF(filename=curr_file, varname='v_rho')

        points = np.array(([0, 0, 0], [np.pi, 1, 0], [2 * np.pi, 0, 0]))
        time = datetime.datetime.now()

        assert all(u.at(points, time) == [1, 1, 1])
        print(np.cos(points[:, 0] / 2) / 2)
        assert all(np.isclose(v.at(points, time), np.cos(points[:, 0] / 2) / 2))

class TestVectorVariable:

    def test_construction(self):
        curr_file = os.path.join(test_dir, 'staggered_sine_channel.nc')
        u = Variable.from_netCDF(filename=curr_file, varname='u_rho')
        v = Variable.from_netCDF(filename=curr_file, varname='v_rho')
        gvp = VectorVariable(name='velocity', units='m/s', time=u.time, variables=[u, v])
        assert gvp.name == 'velocity'
        assert gvp.units == 'm/s'
        assert gvp.varnames[0] == 'u_rho'
#         pytest.set_trace()

    def test_at(self):
        curr_file = os.path.join(test_dir, 'staggered_sine_channel.nc')
        gvp = VectorVariable.from_netCDF(filename=curr_file,
                                         varnames=['u_rho', 'v_rho'])
        points = np.array(([0, 0, 0], [np.pi, 1, 0], [2 * np.pi, 0, 0]))
        time = datetime.datetime.now()

        assert all(np.isclose(gvp.at(points, time)[:, 1], np.cos(points[:, 0] / 2) / 2))


if __name__ == "__main__":
    pass
