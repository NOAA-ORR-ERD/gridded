#!/usr/bin/env python

import pytest
import os
import netCDF4 as nc

from gridded import Dataset
from gridded.tests.utilities import get_test_file_dir

test_dir = get_test_file_dir()
'''
Need to hook this up to existing test data infrastructure
'''
def test_init():
    """ tests you can intitize a basic datset"""
    sinusoid_fn = os.path.join(test_dir, 'staggered_sine_channel.nc')
    sinusoid = Dataset(sinusoid_fn)

