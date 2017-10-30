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
from gridded.grids import Grid_S
from gridded.time import Time

from gridded.depth import S_Depth, L_Depth

test_dir = get_test_file_dir()

@pytest.fixture(scope='module')
def get_s_depth():
    '''
    This is setup for a ROMS S-level Depth that is on a square grid with a center
    mound. Control vars: sz=xy size, center_el=height of the mound in meters,
    d0=general depth in meters, sig=steepness of mound, nz=number of z levels.
    '''
    sz=40
    center_el=10
    d0 = 20
    sig = 0.75
    nz = 11
    node_lat, node_lon = np.mgrid[0:sz,0:sz]
    b_data = np.empty((sz,sz))
    for x in range(0,sz):
        for y in range(0,sz):
            b_data[x,y] = d0 - center_el*np.exp(-0.1*((x-(sz/2))**2 / 2.*((sig)**2) + (y-(sz/2))**2 / 2.*((sig)**2)))
    z_data = np.empty((3,sz,sz))
    for t in range(0,3):
        for x in range(0,sz):
            for y in range(0,sz):
                z_data[t,x,y] = (t - 1.)/2.
    g = Grid_S(node_lon=node_lon, node_lat=node_lat)
    bathy = Variable(name='bathy',
                     grid = g,
                     data = b_data)
    t_data = np.array([Time.constant_time().data[0] + datetime.timedelta(minutes=10*d) for d in range(0,3)])
    zeta = Variable(name='zeta',
                    time=Time(data=t_data),
                    grid=g,
                    data=z_data)


    s_w = np.linspace(-1,0,nz)
    s_rho = (s_w[0:-1] + s_w[1:]) /2
    #equidistant layers, no stretching
    Cs_w = np.linspace(-1,0,nz)
    Cs_w = 1-1/np.exp(2*Cs_w)
    Cs_w /= -Cs_w[0]
    Cs_r = (Cs_w[0:-1] + Cs_w[1:]) /2
    hc = np.array([0,])

    sd = S_Depth(time=zeta.time,
                 grid=zeta.grid,
                 bathymetry=bathy,
                 zeta=zeta,
                 terms={'s_w':s_w,
                        's_rho':s_rho,
                        'Cs_w':Cs_w,
                        'Cs_r':Cs_r,
                        'hc':hc})
    return sd

@pytest.fixture(scope='module')
def get_l_depth():
    '''
    This sets up a HYCOM level depth where surface is index 0 and bottom is last index
    '''

    depth_levels = np.array(([0,1,2,4,6,10]))
    ld = L_Depth(surface_index=0, bottom_index=len(depth_levels)-1, terms={'depth_levels':depth_levels})
    return ld

class Test_S_Depth(object):

    def test_construction(self, get_s_depth):
        assert get_s_depth is not None

    def test_structure(self, get_s_depth):
        sd = get_s_depth
        sd.Cs_w = np.linspace(-1,0,len(sd.Cs_w))
        sd.Cs_r = (sd.Cs_w[0:-1] + sd.Cs_w[1:]) /2

        sz = sd.bathymetry.data.shape[1]
        levels = sd.get_section(sd.time.data[1], 'w')[:,:,:] #3D cross section
        edge = np.linspace(20,0, len(sd.Cs_w))
        center = np.linspace(10,0,len(sd.Cs_w))
        assert np.allclose(levels[:,0,0], edge)
        assert np.allclose(levels[:, int(sz/2), int(sz/2)], center)

    def test_interpolation_alphas(self, get_s_depth):
        sd = get_s_depth


class Test_L_Depth(object):

    def test_construction(self, get_l_depth):
        assert get_l_depth is not None

    def test_interpolation_alphas(self, get_l_depth):
        ld = get_l_depth
        points = np.array(([0,0,0],[1,1,0]))
        idxs, alphas = ld.interpolation_alphas(points)
        assert idxs is None
        assert alphas is None
        points = np.array(([0,0,0],[1,1,0.5],[0,0,1]))
        idxs, alphas = ld.interpolation_alphas(points)
        assert np.all(idxs == np.array([-1,0,1]))
        assert np.all(np.isclose(alphas, np.array([-1,0.5,0])))
        points = np.array(([0,0,-1],[0,0,10]))
        idxs, alphas = ld.interpolation_alphas(points)
        assert np.all(idxs == np.array([-1,-1]))
        assert np.all(alphas == np.array([-1,-1]))