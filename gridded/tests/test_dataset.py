#!/usr/bin/env python

from __future__ import absolute_import, division, print_function, unicode_literals

import pytest
import os
import netCDF4 as nc

from gridded import Dataset
from .utilities import get_test_file_dir

test_dir = get_test_file_dir()

# Need to hook this up to existing test data infrastructure
# and add more infrustructure...

sample_sgrid_file = os.path.join(test_dir, 'staggered_sine_channel.nc')


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

# def test_get_variables_by_attribute():
#     gds = Dataset(sample_sgrid_file)

#     print(gds.varibles)

#     assert False


def test_save_invalid_format():
    ds = Dataset()

    with pytest.raises(ValueError):
        ds.save("a_filename.txt", format="text")


