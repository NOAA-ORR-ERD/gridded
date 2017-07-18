#!/usr/bin/env python

# from __future__ import absolute_import, division, print_function, unicode_literals

# import pytest
import os
import netCDF4 as nc

from gridded import Dataset
from gridded.tests.utilities import get_test_file_dir

test_dir = get_test_file_dir()

# Need to hook this up to existing test data infrastructure
# and add more infrustructure...


# def test_init():
#     """ tests you can intitilize a basic datset"""
#     sinusoid_fn = os.path.join(test_dir, 'staggered_sine_channel.nc')
#     sinusoid = Dataset(sinusoid_fn)


# def test_get_variables_by_attribute():
#     ds = Dataset(os.path.join(test_dir, 'staggered_sine_channel.nc'))

#     print(ds)

#     assert False

