#!/usr/bin/env python

import pytest
import os
import netCDF4 as nc

from gridded import Dataset

base_dir = os.path.dirname(__file__)
'''
Need to hook this up to existing test data infrastructure
'''

s_data = os.path.join(base_dir, 'test_data')

roms_fn = os.path.join(s_data, 'sgrid_roms.nc')
# gen_all(path=s_data)
# circular_fn = os.path.join(s_data, 'circular_3D.nc')
# circular = nc.Dataset(circular_fn)

sinusoid_fn = os.path.join(s_data, 'staggered_sine_channel.nc')
sinusoid = nc.Dataset(sinusoid_fn)

# circular_3D = os.path.join(s_data, '3D_circular.nc')
# circular_3D = nc.Dataset(circular_3D)

tri_ring_fn = os.path.join(s_data, 'tri_ring.nc')
tri_ring = nc.Dataset(tri_ring_fn)

def test_init():
    """ tests you can intitize a basic datset"""
    D = Dataset(sinusoid_fn)
    return D

if __name__ == '__main__':
    a = test_init()
    pass
