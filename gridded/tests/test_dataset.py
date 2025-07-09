#!/usr/bin/env python


import pytest
import os
import netCDF4 as nc

from gridded import Dataset
from gridded.grids import Grid_S
from .utilities import TEST_DATA


# Need to hook this up to existing test data infrastructure
# ... and add more infrastructure

sample_sgrid_file = TEST_DATA / 'staggered_sine_channel.nc'
arakawa_c_file = TEST_DATA / 'arakawa_c_test_grid.nc'


def test_load_sgrid():
    """ tests you can initialize an conforming sgrid file"""
    sinusoid = Dataset.from_netCDF(sample_sgrid_file)

    assert isinstance(sinusoid.grid, Grid_S)

    assert True  # just to make it a test


def test_init_from_netcdf_file_directly():
    """
    This should raise a deprecation warning, but still work
    """
    with pytest.warns(DeprecationWarning):
        gds = Dataset(arakawa_c_file)

    print(gds.info)

    assert isinstance(gds.grid, Grid_S)
    assert len(gds.variables) == 6


def test_info():
    """
    Make sure the info property is working
    This doesn't test much -- jsut tht it won't crash
    """
    gds = Dataset.from_netCDF(sample_sgrid_file)

    info = gds.info

    print(info)
    # just a couple checks to make sure it's not totally bogus
    assert "gridded.Dataset:" in info
    assert "variables:" in info
    assert "attributes:" in info

def test_get_variable_by_attribute_one_there():
    gds = Dataset.from_netCDF(arakawa_c_file)

    vars = gds.get_variables_by_attribute('long_name', 'v-momentum component')

    assert len(vars) == 1
    assert vars[0].attributes['long_name'] == 'v-momentum component'

def test_get_variable_by_attribute_multiple():
    gds = Dataset.from_netCDF(arakawa_c_file)

    vars = gds.get_variables_by_attribute('units', 'meter second-1')


    assert len(vars) == 2
    assert vars[0].attributes['units'] == 'meter second-1'
    assert vars[1].attributes['units'] == 'meter second-1'


def test_get_variable_by_attribute_not_there():
    """
    This should return an empty list
    """
    gds = Dataset.from_netCDF(arakawa_c_file)

    var = gds.get_variables_by_attribute('some_junk', 'more_junk')

    assert var == []


def test_save_invalid_format():
    ds = Dataset()

    with pytest.raises(ValueError):
        ds.save("a_filename.txt", format="text")


