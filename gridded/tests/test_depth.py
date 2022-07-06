#!/usr/bin/env python

# py2/3 compatibility
from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
)

import os
import sys
import datetime

import pytest
import numpy as np
import netCDF4 as nc

from gridded.variable import Variable, VectorVariable
from gridded.tests.utilities import get_test_file_dir
from gridded.grids import Grid_S
from gridded.time import Time

from gridded.depth import S_Depth, L_Depth

test_dir = get_test_file_dir()


@pytest.fixture(scope="module")
def get_s_depth():
    """
    This is setup for a ROMS S-level Depth that is on a square grid with a center
    mound. Control vars: sz=xy size, center_el=height of the mound in meters,
    d0=general depth in meters, sig=steepness of mound, nz=number of z levels.
    """
    sz = 40
    center_el = 10
    d0 = 20
    sig = 0.75
    nz = 11
    node_lat, node_lon = np.mgrid[0:sz, 0:sz]
    b_data = np.empty((sz, sz))
    for x in range(0, sz):
        for y in range(0, sz):
            b_data[x, y] = d0 - center_el * np.exp(
                -0.1
                * (
                    (x - (sz / 2)) ** 2 / 2.0 * ((sig) ** 2)
                    + (y - (sz / 2)) ** 2
                    / 2.0
                    * ((sig) ** 2)
                )
            )
    z_data = np.empty((3, sz, sz))
    for t in range(0, 3):
        for x in range(0, sz):
            for y in range(0, sz):
                z_data[t, x, y] = (t - 1.0) / 2.0
    g = Grid_S(node_lon=node_lon, node_lat=node_lat)
    bathy = Variable(name="bathy", grid=g, data=b_data)
    t_data = np.array(
        [
            Time.constant_time().data[0]
            + datetime.timedelta(minutes=10 * d)
            for d in range(0, 3)
        ]
    )
    zeta = Variable(
        name="zeta",
        time=Time(data=t_data),
        grid=g,
        data=z_data,
    )

    s_w = np.linspace(-1, 0, nz)
    s_rho = (s_w[0:-1] + s_w[1:]) / 2
    # equidistant layers, no stretching
    Cs_w = np.linspace(-1, 0, nz)
    Cs_w = 1 - 1 / np.exp(2 * Cs_w)
    Cs_w /= -Cs_w[0]
    Cs_r = (Cs_w[0:-1] + Cs_w[1:]) / 2
    hc = np.array([0])

    sd = S_Depth(
        time=zeta.time,
        grid=zeta.grid,
        bathymetry=bathy,
        zeta=zeta,
        terms={
            "s_w": s_w,
            "s_rho": s_rho,
            "Cs_w": Cs_w,
            "Cs_r": Cs_r,
            "hc": hc,
        },
    )
    return sd


@pytest.fixture(scope="module")
def get_l_depth():
    """
    This sets up a HYCOM level depth where surface is index 0 and bottom is last index
    """

    depth_levels = np.array(([0, 1, 2, 4, 6, 10]))
    ld = L_Depth(
        surface_index=0,
        bottom_index=len(depth_levels) - 1,
        terms={"depth_levels": depth_levels},
    )
    return ld


class Test_S_Depth(object):
    def test_construction(self, get_s_depth):
        assert get_s_depth is not None

    def test_structure(self, get_s_depth):
        sd = get_s_depth
        sd.Cs_w = np.linspace(-1, 0, len(sd.Cs_w))
        sd.Cs_r = (sd.Cs_w[0:-1] + sd.Cs_w[1:]) / 2

        sz = sd.bathymetry.data.shape[1]
        levels = sd.get_section(sd.time.data[1], "w")[
            :, :, :
        ]  # 3D cross section
        edge = np.linspace(20, 0, len(sd.Cs_w))
        center = np.linspace(10, 0, len(sd.Cs_w))
        assert np.allclose(levels[:, 0, 0], edge)
        assert np.allclose(
            levels[:, int(sz / 2), int(sz / 2)], center
        )

    def test_interpolation_alphas_bottom(self, get_s_depth):
        '''
        We will focus on the center mound to see if the correct alphas and
        indices are returned for various points nearby.
        interpolation_alphas(self, points, time, data_shape, _hash=None, extrapolate=False):
        '''
        sd = get_s_depth
        # query center mound (20,20) at 3 depths. 1st point is 0.1m off the seafloor and 10% of
        # the distance to the next s-layer (meaning, expected alpha returned is 10%)
        # 2nd point is directly on the seafloor (should register as 'in bounds' and 0 alpha)
        # 3rd point is 0.1m underground, and should indicate with -2 alpha
        points = np.array([[20,20, 9.9],[20,20,10.0], [20,20,10.1]])
        ts = sd.time.data[1]
        idx, alphas = sd.interpolation_alphas(points, ts, [sd.num_w_levels,])
        assert sd.zero_ref == 'surface'
        assert np.all(idx == [1,1,-1])
        assert np.all(np.isclose(alphas, [0.1, 0, -2]))
        
        # since zeta is 0 in this case, the behavior should be the same as in absolute mode
        sd.zero_ref = 'absolute'
        idx, alphas = sd.interpolation_alphas(points, ts, [sd.num_w_levels,])
        assert np.all(idx == [1,1,-1])
        assert np.all(np.isclose(alphas, [0.1, 0, -2]))

        # time.data[2] changes the zeta to 0.5 (.5m up), increasing the total water column at the query point
        # to 10.5m. Since zero_ref is still 'surface', the depth of each particle is effectively
        # decreased by zeta. This test may very well change since this behavior can be problematic. 
        ts = sd.time.data[2]
        sd.zero_ref = 'surface'
        idx, alphas = sd.interpolation_alphas(points, ts, [sd.num_w_levels,])
        assert np.all(idx == [1,1,1])
        levels = sd.get_section(ts, 'w')[:,20,20]
        zetas = sd.zeta.at(points, ts)
        for alph, dep, zetas in zip(alphas, points[:,2], zetas):
            expected = 1 - ((dep - zetas) - levels[1]) / (levels[0] - levels[1])
            assert np.isclose(alph, expected)
        # assert np.all(np.isclose(alphas, [0.1, 0, -2]))
        
        # since zeta is non-zero, behavior will be different. It should be similar to the first case
        # but not identical since the layers have been stretched slightly differently due to zeta
        sd.zero_ref = 'absolute'
        idx, alphas = sd.interpolation_alphas(points, ts, [sd.num_w_levels,])
        assert np.all(idx == [1,1,-1])
        levels = sd.get_section(ts, 'w')[:,20,20]
        zetas = sd.zeta.at(points, ts)
        expected = 1 - (points[0,2] - levels[1]) / (levels[0] - levels[1])
        assert np.isclose(alphas[0], expected)
        assert np.all(np.isclose(alphas[1:], [0,-2]))
    
    def test_interpolation_alphas_surface(self, get_s_depth):
        sd = get_s_depth
        sd.zero_ref = 'surface'
        points = np.array([[20,20, 0],[20,20,0.1], [20,20,-0.1]])
        ts = sd.time.data[1]
        idx, alphas = sd.interpolation_alphas(points, ts, [sd.num_w_levels,])
        # only the element 0.1m deep should register with an index and alpha since
        # it is the only element below surface
        assert sd.zero_ref == 'surface'
        assert np.all(idx == [-1,10,-1]) 
        assert np.all(alphas == [-1,0.9,-1])

        # switch to timestep with -0.5m zeta
        ts = sd.time.data[0]
        idx, alphas = sd.interpolation_alphas(points, ts, [sd.num_w_levels,])
        assert np.all(idx == [-1,10,-1])
        # on surface and above should be the same. slightly below surface point should have
        # a slightly lower alpha due to slightly compressed coordinate compared to original
        # test.
        assert np.all(alphas[[0,2]] == [-1,-1])
        assert alphas[1] < 0.9 and alphas[1] > 0.8

        sd.zero_ref = 'absolute'
        idx, alphas = sd.interpolation_alphas(points, ts, [sd.num_w_levels,])
        # this combo of depths and zeta and zero ref should evaluate all depths
        # as 'above the surface' and so idx, alphas should be None, None
        assert idx == None and alphas == None


class Test_L_Depth(object):
    def test_construction(self, get_l_depth):
        assert get_l_depth is not None

    def test_interpolation_alphas(self, get_l_depth):
        ld = get_l_depth
        points = np.array(([0, 0, 0], [1, 1, 0]))
        idxs, alphas = ld.interpolation_alphas(points)
        assert idxs is None
        assert alphas is None
        points = np.array(([0, 0, 0],
                           [1, 1, 0.5],
                           [0, 0, 1],
                           ))
        idxs, alphas = ld.interpolation_alphas(points)
        assert np.all(idxs == np.array([-1, 0, 1]))
        assert np.all(
            np.isclose(alphas, np.array([-1, 0.5, 0]))
        )
        points = np.array(([0, 0, -1], [0, 0, 10]))
        idxs, alphas = ld.interpolation_alphas(points)
        assert np.all(idxs == np.array([-1, -1]))
        assert np.all(alphas == np.array([-1, -1]))
