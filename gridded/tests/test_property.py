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

from gridded.grids import Grid

base_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(base_dir, 'sample_data'))


'''
Need to hook this up to existing test data infrastructure
'''

s_data = os.path.join(base_dir, 'test_data')
# gen_all(path=s_data)

sinusoid = os.path.join(s_data, 'staggered_sine_channel.nc')
sinusoid = nc.Dataset(sinusoid)

# circular_3D = os.path.join(s_data, '3D_circular.nc')
# circular_3D = nc.Dataset(circular_3D)

tri_ring = os.path.join(s_data, 'tri_ring.nc')
tri_ring = nc.Dataset(tri_ring)


# class TestS_Depth_T1:
#
#     def test_construction(self):
#
#         test_grid = PyGrid_S(node_lon=np.array([[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]),
#                             node_lat=np.array([[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]))
#
#         u = np.zeros((3, 4, 4), dtype=np.float64)
#         u[0, :, :] = 0
#         u[1, :, :] = 1
#         u[2, :, :] = 2
#
#         w = np.zeros((4, 4, 4), dtype=np.float64)
#         w[0, :, :] = 0
#         w[1, :, :] = 1
#         w[2, :, :] = 2
#         w[3, :, :] = 3
#
#         bathy_data = -np.array([[1, 1, 1, 1],
#                                [1, 2, 2, 1],
#                                [1, 2, 2, 1],
#                                [1, 1, 1, 1]], dtype=np.float64)
#
#         Cs_w = np.array([1.0, 0.6667, 0.3333, 0.0])
#         s_w = np.array([1.0, 0.6667, 0.3333, 0.0])
#         Cs_r = np.array([0.8333, 0.5, 0.1667])
#         s_rho = np.array([0.8333, 0.5, 0.1667])
#         hc = np.array([1])
#
#         b = Bathymetry(name='bathymetry', data=bathy_data, grid=test_grid, time=None)
#
#         dep = S_Depth_T1(bathymetry=b, terms=dict(zip(S_Depth_T1.default_terms[0], [Cs_w, s_w, hc, Cs_r, s_rho])), dataset='dummy')
#         assert dep is not None
#
#         corners = np.array([[0, 0, 0], [0, 3, 0], [3, 3, 0], [3, 0, 0]], dtype=np.float64)
#         res, alph = dep.interpolation_alphas(corners, w.shape)
#         assert res is None  # all particles on surface
#         assert alph is None  # all particles on surface
#         res, alph = dep.interpolation_alphas(corners, u.shape)
#         assert res is None  # all particles on surface
#         assert alph is None  # all particles on surface
#
#         pts2 = corners + (0, 0, 2)
#         res = dep.interpolation_alphas(pts2, w.shape)
#         assert all(res[0] == 0)  # all particles underground
#         assert np.allclose(res[1], -2.0)  # all particles underground
#         res = dep.interpolation_alphas(pts2, u.shape)
#         assert all(res[0] == 0)  # all particles underground
#         assert np.allclose(res[1], -2.0)  # all particles underground
#
#         layers = np.array([[0.5, 0.5, .251], [1.5, 1.5, 1.0], [2.5, 2.5, 1.25]])
#         res, alph = dep.interpolation_alphas(layers, w.shape)
#         print res
#         print alph
#         assert all(res == [3, 2, 1])
#         assert np.allclose(alph, np.array([0.397539, 0.5, 0]))



'''
Analytical cases:

Triangular
    grid shape: (nodes = nv, faces = nele)
    data_shapes: (time, depth, nv),
                 (time, nv),
                 (depth, nv),
                 (nv)
    depth types: (None),
                 (constant),
                 (sigma v1),
                 (sigma v2),
                 (levels)
    test points: 2D surface (time=None, depth=None)
                     - nodes should be valid
                     - off grid should extrapolate with fill value or Error
                     - interpolation elsewhere
                 2D surface (time=t, depth=None)
                     - as above, validate time interpolation



Quad
    grid shape: (nodes:(x,y))
                (nodes:(x,y), faces(xc, yc))
                (nodes:(x,y), faces(xc, yc), edge1(x, yc), edge2(xc, y))
    data_shapes: (time, depth, x, y),
                 (time, x, y),
                 (depth, x, y),
                 (x,y)
    depth types: (None),
                 (constant),
                 (sigma v1),
                 (sigma v2),
                 (levels)

'''


class TestGriddedProp:


    def test_construction(self):

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

        curr_file = os.path.join(s_data, 'staggered_sine_channel.nc')
        k = Variable.from_netCDF(filename=curr_file, varname='u', name='u')
        assert k.name == u.name
        assert k.units == 'm/s'
        # fixme: this was failing
        # assert k.time == u.time
        assert k.data[0, 0] == u.data[0, 0]

    def test_at(self):
        curr_file = os.path.join(s_data, 'staggered_sine_channel.nc')
        u = Variable.from_netCDF(filename=curr_file, varname='u_rho')
        v = Variable.from_netCDF(filename=curr_file, varname='v_rho')

        points = np.array(([0, 0, 0], [np.pi, 1, 0], [2 * np.pi, 0, 0]))
        time = datetime.datetime.now()

        assert all(u.at(points, time) == [1, 1, 1])
        print(np.cos(points[:, 0] / 2) / 2)
        assert all(np.isclose(v.at(points, time), np.cos(points[:, 0] / 2) / 2))

class TestGridVectorProp:

    def test_construction(self):
        curr_file = os.path.join(s_data, 'staggered_sine_channel.nc')
        u = Variable.from_netCDF(filename=curr_file, varname='u_rho')
        v = Variable.from_netCDF(filename=curr_file, varname='v_rho')
        gvp = VectorVariable(name='velocity', units='m/s', time=u.time, variables=[u, v])
        assert gvp.name == 'velocity'
        assert gvp.units == 'm/s'
        assert gvp.varnames[0] == 'u_rho'
#         pytest.set_trace()

    def test_at(self):
        curr_file = os.path.join(s_data, 'staggered_sine_channel.nc')
        gvp = VectorVariable.from_netCDF(filename=curr_file,
                                         varnames=['u_rho', 'v_rho'])
        points = np.array(([0, 0, 0], [np.pi, 1, 0], [2 * np.pi, 0, 0]))
        time = datetime.datetime.now()

        assert all(np.isclose(gvp.at(points, time)[:, 1], np.cos(points[:, 0] / 2) / 2))


if __name__ == "__main__":
    pass
