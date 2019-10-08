from __future__ import (absolute_import, division, print_function)

import os
import contextlib

import pytest

import numpy as np
from netCDF4 import Dataset

from gridded.pyugrid.ugrid import UGrid
from gridded.pyugrid.grid_io import load_from_varnames


# @pytest.fixture
@contextlib.contextmanager
def non_compliant_mesh(fname):
    """
    Dummy file based on:
    https://gnome.orr.noaa.gov/py_gnome_testdata/COOPSu_CREOFS.nc

    """
    nc = Dataset(fname, 'w', diskless=True)
    nc.grid_type = 'Triangular'
    nc.createDimension('nbi', 4)
    nc.createDimension('three', 3)
    nc.createDimension('nbnd', 5443)
    nc.createDimension('node', 74061)
    nc.createDimension('nele', 142684)

    bnd = nc.createVariable('bnd', 'i4', dimensions=('nbnd', 'nbi'))
    bnd[:] = np.random.random((5443, 4))

    lon = nc.createVariable('lon', 'f4', dimensions=('node'))
    lon[:] = np.random.random((74061))

    lat = nc.createVariable('lat', 'f4', dimensions=('node'))
    lat[:] = np.random.random((74061))

    nbe = nc.createVariable('nbe', 'i4', dimensions=('three', 'nele'))
    nbe.order = 'ccw'
    nbe[:] = np.random.random((3, 142684))

    nv = nc.createVariable('nv', 'i4', dimensions=('three', 'nele'))
    nv[:] = np.random.random((3, 142684))
    try:
        yield nc
    finally:
        nc.close()

def test_load_from_varnames_good_mapping():
    mapping = {'attribute_check': ('grid_type', 'triangular'),
               'faces': 'nv',
               'nodes_lon': 'lon',
               'nodes_lat': 'lat',
               'boundaries': 'bnd',
               'face_face_connectivity': 'nbe'}

    fname = 'non_compliant_ugrid.nc'
    with non_compliant_mesh(fname):
        ug = load_from_varnames(fname, mapping)
    assert isinstance(ug, UGrid)


def test_load_from_varnames_bad_mapping():
    mapping = {'attribute_check': ('grid_type', 'triangular'),
               'faces': 'nv',
               'nodes_lon': 'longitude',
               'nodes_lat': 'latitude',
               'boundaries': 'bnd',
               'face_face_connectivity': 'nbe'}

    fname = 'non_compliant_ugrid.nc'
    with non_compliant_mesh(fname):
        with pytest.raises(KeyError):
            load_from_varnames(fname, mapping)
