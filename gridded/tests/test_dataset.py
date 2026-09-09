#!/usr/bin/env python


import datetime
import os

from gridded.time import Time
import netCDF4 as nc
import pytest

from gridded import Dataset
from gridded.grids import Grid_S

from .utilities import TEST_DATA

# Need to hook this up to existing test data infrastructure
# ... and add more infrastructure

sample_sgrid_file = TEST_DATA / "staggered_sine_channel.nc"
arakawa_c_file = TEST_DATA / "arakawa_c_test_grid.nc"


def test_load_sgrid():
    """tests you can initialize an conforming sgrid file"""
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

    vars = gds.get_variables_by_attribute("long_name", "v-momentum component")

    assert len(vars) == 1
    assert vars[0].attributes["long_name"] == "v-momentum component"


def test_get_variable_by_attribute_multiple():
    gds = Dataset.from_netCDF(arakawa_c_file)

    vars = gds.get_variables_by_attribute("units", "meter second-1")

    assert len(vars) == 2
    assert vars[0].attributes["units"] == "meter second-1"
    assert vars[1].attributes["units"] == "meter second-1"


def test_get_variable_by_attribute_not_there():
    """
    This should return an empty list
    """
    gds = Dataset.from_netCDF(arakawa_c_file)

    var = gds.get_variables_by_attribute("some_junk", "more_junk")

    assert var == []


def test_diff_method():
    ds1 = Dataset.from_netCDF(arakawa_c_file)
    ds2 = Dataset.from_netCDF(arakawa_c_file)

    diff = ds1._diff(ds2)
    assert diff is None

    # Modify ds2 to create a difference
    ds2.attributes = {"name": "modified_name"}
    diff = ds1._diff(ds2)
    assert diff is not None
    assert diff['self']['attributes'] == "self.attributes: {}, other.attributes: {'name': 'modified_name'}"

def test_eq_method():
    ds1 = Dataset.from_netCDF(arakawa_c_file)
    ds2 = Dataset.from_netCDF(arakawa_c_file)

    assert ds1 == ds2

    ds2.attributes = {"name": "modified_name"}
    assert ds1 != ds2


def test_save_invalid_format():
    ds = Dataset()

    with pytest.raises(ValueError):
        ds.save("a_filename.txt", format="text")

# def test_save_basic():
#     ds = Dataset()
#     ds.name = "a_test_dataset"

#     filename = "a_test_dataset.nc"
#     ds.save(filename)

#     assert os.path.exists(filename)

#     # Clean up
#     os.remove(filename)

# def test_save_with_time():
#     ds = Dataset()
#     ds.name = "a_test_dataset"
#     ds.time = Time(data=[datetime.datetime.now()])

#     filename = "a_test_dataset.nc"
#     ds.save(filename)

#     assert os.path.exists(filename)
    
#     ds2 = Dataset.load(filename)
#     assert ds.name == "a_test_dataset"
#     assert ds2 == ds

#     # Clean up
#     os.remove(filename)
