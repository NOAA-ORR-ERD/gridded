"""
testing verdat read/write capability
"""

import os
import gridded
from gridded.io import verdat

test_filename = os.path.join(os.path.split(__file__)[0], "example_verdat.verdat")
test_filename_no_units = os.path.join(os.path.split(__file__)[0], "example_verdat_no_units.verdat")


def test_read():
    """
    at least it does something
    """
    ds = verdat.dataset_from_verdat(test_filename)

    assert isinstance(ds, gridded.Dataset)


def test_read_no_units():
    ds = verdat.dataset_from_verdat(test_filename_no_units)

    assert ds.variables['depth'].attrs['units'] == ""



