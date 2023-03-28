#!/usr/bin/env python


import pytest
import os
import netCDF4 as nc

from gridded import Dataset
from .utilities import get_test_file_dir

test_dir = get_test_file_dir()

# Need to hook this up to existing test data infrastructure
# ... and add more infrastructure

sample_sgrid_file = os.path.join(test_dir, 'staggered_sine_channel.nc')
arakawa_c_file = os.path.join(test_dir, 'arakawa_c_test_grid.nc')


def test_load_sgrid():
    """ tests you can intitilize an conforming sgrid file"""
    sinusoid = Dataset(sample_sgrid_file)

    assert True  # just to make it a test


def test_info():
    """
    Make sure the info property is working
    This doesn't test much -- jsut tht it won't crash
    """
    gds = Dataset(sample_sgrid_file)

    info = gds.info

    print(info)
    # just a couple checks to make sure it's not totally bogus
    assert "gridded.Dataset:" in info
    assert "variables:" in info
    assert "attributes:" in info

def test_get_variable_by_attribute_one_there():
    gds = Dataset(arakawa_c_file)

    vars = gds.get_variables_by_attribute('long_name', 'v-momentum component')

    assert len(vars) == 1
    assert vars[0].attributes['long_name'] == 'v-momentum component'

def test_get_variable_by_attribute_multiple():
    gds = Dataset(arakawa_c_file)

    vars = gds.get_variables_by_attribute('units', 'meter second-1')


    assert len(vars) == 2
    assert vars[0].attributes['units'] == 'meter second-1'
    assert vars[1].attributes['units'] == 'meter second-1'


def test_get_variable_by_attribute_not_there():
    """
    This should return an empty list
    """
    gds = Dataset(arakawa_c_file)

    var = gds.get_variables_by_attribute('some_junk', 'more_junk')

    assert var == []


def test_save_invalid_format():
    ds = Dataset()

    with pytest.raises(ValueError):
        ds.save("a_filename.txt", format="text")


