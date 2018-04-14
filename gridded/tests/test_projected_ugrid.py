#!/usr/bin/env python

"""
tests loading a UGRID file with projected coords

this test uses a data file auto-downloaded from:

"""

from gridded import Dataset

data_file = "temp_data/projected_coords_ugrid.nc"


def test_load():
    """
    the file should load without error
    """
    ds = Dataset(data_file)

    assert True

