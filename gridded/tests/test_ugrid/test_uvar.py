#!/usr/bin/env python

"""
Tests for the UVar object
"""


import os

import numpy as np
import pytest

from gridded.pyugrid.uvar import UVar
from .utilities import chdir

# pytestmark = pytest.mark.skipif(True, reason="gridded does not support UVars anymore")


test_files = os.path.join(os.path.dirname(__file__), 'files')


def test_init():
    d = UVar('depth', location='node', data=[1.0, 2.0, 3.0, 4.0])

    assert d.name == 'depth'
    assert np.array_equal(d.data, [1.0, 2.0, 3.0, 4.0])
    assert d.location == 'node'
    assert d.attributes == {}

    with pytest.raises(ValueError):
        d = UVar('depth', location='nodes')


def test_add_data():
    d = UVar('depth', location='node')
    assert d.name == 'depth'
    assert np.array_equal(d.data, [])

    # Add the data:
    d.data = [1.0, 2.0, 3.0, 4.0]

    assert np.array_equal(d.data, [1.0, 2.0, 3.0, 4.0])
    # Duck type check of nd.ndarray.
    d.data *= 2
    assert np.array_equal(d.data, [2.0, 4.0, 6.0, 8.0])


def test_delete_data():
    d = UVar('depth', location='node', data=[1.0, 2.0, 3.0, 4.0])

    del d.data
    assert np.array_equal(d.data, [])


def test_str():
    d = UVar('depth', location='node', data=[1.0, 2.0, 3.0, 4.0])
    assert str(d) == ('UVar object: depth, on the nodes, and 4 data points\n'
                      'Attributes: {}')


def test_add_attributes():
    d = UVar('depth', location='node', data=[1.0, 2.0, 3.0, 4.0])
    d.attributes = {'standard_name': 'sea_floor_depth_below_geoid',
                    'units': 'm',
                    'positive': 'down'}
    assert d.attributes['units'] == 'm'
    assert d.attributes['positive'] == 'down'


def test_nc_variable():
    """
    test that it works with a netcdf variable object
    """
    import netCDF4

    # make a variable
    with chdir(test_files):
        fname = 'junk.nc'
        ds = netCDF4.Dataset(fname, mode='w')
        ds.createDimension('dim', (10))
        var = ds.createVariable('a_var', float, ('dim'))
        var[:] = np.arange(10)
        # give it some attributes
        var.attr_1 = 'some value'
        var.attr_2 = 'another value'

        # make a UVar from it
        uvar = UVar("a_var", 'node', data=var)

        assert uvar._data is var  # preserved the netcdf variable
        print(uvar.attributes)
        assert uvar.attributes == {'attr_1': 'some value',
                                   'attr_2': 'another value'}
        # access the data
        assert np.array_equal(uvar[3:5], [3.0, 4.0])
